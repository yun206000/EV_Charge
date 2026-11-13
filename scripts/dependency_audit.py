import argparse
import csv
import json
import math
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from packaging.version import Version

"""
Dependency vulnerability scan, step 2 (step 1 is pip-audit in scripts/dependency_audit.sh): rate every vulnerability
found by pip-audit and write

    <out-dir>/dependency_audit.csv          one row per vulnerability (package, version, id, aliases, severity, fix)
    <out-dir>/dependency_audit_summary.txt  one line per vulnerable package
    dependency-audit-sonar.json              SonarQube generic external issues: one issue per vulnerable package, on
                                             the line that pins it (docker/requirements-*.txt, environment.yml or the
                                             Dockerfile), read by sonar-project.properties

Severity: the GitHub advisory rating (CRITICAL / HIGH / MODERATE / LOW) of the vulnerability or of its GHSA alias in
the OSV database (api.osv.dev); without one, the CVSS 3.x base score of the OSV record; otherwise UNRATED.

    python scripts/dependency_audit.py --audit Results/Security/pip_audit.json
"""

OSV_URL = "https://api.osv.dev/v1/vulns/{}"
LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNRATED"]
SONAR_SEVERITY = {"CRITICAL": "BLOCKER", "HIGH": "HIGH", "MEDIUM": "MEDIUM", "LOW": "LOW", "UNRATED": "MEDIUM"}
# where a package is pinned: (file, pattern of the pinning line); names are compared in normalised form
PIN_FILES = [("docker/requirements-pcrl.txt", r"^{name}=="), ("docker/requirements-test.txt", r"^{name}=="),
             ("docker/Dockerfile", r"\b{name}=="),
             # legacy environment (IMAGE=pcrl:1.0)
             ("docker/legacy-py38/requirements-pcrl.txt", r"^{name}=="),
             ("docker/legacy-py38/requirements-test.txt", r"^{name}=="),
             ("docker/legacy-py38/environment.yml", r"^\s*-\s*{name}(-python)?="),
             ("docker/legacy-py38/Dockerfile", r"\b{name}==")]
FALLBACK = ("docker/Dockerfile", 1)
MESSAGE_LIMIT = 1000

# CVSS 3.x base score (FIRST specification, section 7)
CVSS_WEIGHTS = {"AV": {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}, "AC": {"L": 0.77, "H": 0.44},
                "UI": {"N": 0.85, "R": 0.62}, "C": {"H": 0.56, "L": 0.22, "N": 0}}
PR_WEIGHTS = {"U": {"N": 0.85, "L": 0.62, "H": 0.27}, "C": {"N": 0.85, "L": 0.68, "H": 0.5}}


def normalise(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def roundup(value):
    """CVSS 3.1 Roundup: smallest number with one decimal >= value"""
    integer = int(round(value * 100000))
    return integer / 100000.0 if integer % 10000 == 0 else (math.floor(integer / 10000) + 1) / 10.0


def cvss3_score(vector):
    metrics = dict(part.split(":") for part in vector.split("/")[1:])
    impact_sub = 1 - ((1 - CVSS_WEIGHTS["C"][metrics["C"]]) * (1 - CVSS_WEIGHTS["C"][metrics["I"]]) *
                      (1 - CVSS_WEIGHTS["C"][metrics["A"]]))
    changed = metrics["S"] == "C"
    impact = 7.52 * (impact_sub - 0.029) - 3.25 * (impact_sub - 0.02) ** 15 if changed else 6.42 * impact_sub
    exploitability = (8.22 * CVSS_WEIGHTS["AV"][metrics["AV"]] * CVSS_WEIGHTS["AC"][metrics["AC"]] *
                      PR_WEIGHTS[metrics["S"]][metrics["PR"]] * CVSS_WEIGHTS["UI"][metrics["UI"]])
    if impact <= 0:
        return 0.0
    return roundup(min(1.08 * (impact + exploitability), 10)) if changed else roundup(min(impact + exploitability, 10))


def level_of_score(score):
    if score >= 9:
        return "CRITICAL"
    if score >= 7:
        return "HIGH"
    if score >= 4:
        return "MEDIUM"
    return "LOW" if score > 0 else "UNRATED"


def ghsa_level(record):
    rating = (record.get("database_specific") or {}).get("severity")
    return {"MODERATE": "MEDIUM"}.get(rating, rating) if rating in ["CRITICAL", "HIGH", "MODERATE", "LOW"] else None


def cvss_level(record):
    for severity in record.get("severity") or []:
        if severity.get("type") == "CVSS_V3" and severity.get("score", "").startswith("CVSS:3"):
            return level_of_score(cvss3_score(severity["score"]))
    return None


def fetch_osv(vuln_id):
    try:
        with urllib.request.urlopen(OSV_URL.format(vuln_id), timeout=30) as response:
            return json.load(response)
    except Exception:  # unknown id or no connection: the vulnerability stays unrated
        return {}


def rate(vuln, fetch=None):
    """severity of one pip-audit vulnerability: GHSA rating of the record or its GHSA alias, else CVSS 3.x"""
    fetch = fetch or fetch_osv
    record = fetch(vuln["id"])
    level = ghsa_level(record)
    if level is None:
        aliases = set(vuln.get("aliases") or []) | set(record.get("aliases") or [])
        for alias in sorted(a for a in aliases if a.startswith("GHSA-")):
            level = ghsa_level(fetch(alias))
            if level:
                break
    return level or cvss_level(record) or "UNRATED"


def fix_version(vulns):
    """lowest version that has a fix for every vulnerability with a fix (None: none of them has a fix)"""
    needed = [min(Version(v) for v in vuln["fix_versions"]) for vuln in vulns if vuln.get("fix_versions")]
    return str(max(needed)) if needed else None


def pin_location(name, root="."):
    for file_name, pattern in PIN_FILES:
        path = os.path.join(root, file_name)
        if not os.path.exists(path):
            continue
        regex = re.compile(pattern.format(name=re.escape(normalise(name)).replace("\\-", "[-_.]")), re.IGNORECASE)
        with open(path) as file:
            for number, line in enumerate(file, 1):
                if regex.search(line):
                    return file_name, number
    return FALLBACK


def package_message(name, version, rows):
    counts = [(level, sum(row["severity"] == level for row in rows)) for level in LEVELS]
    without_fix = sum(not row["fix_versions"] for row in rows)
    text = "{} {}: {} known {} ({})".format(
        name, version, len(rows), "vulnerability" if len(rows) == 1 else "vulnerabilities", ", ".join("{} {}".format(n, level.lower()) for level, n in counts if n))
    fix = fix_version([dict(fix_versions=row["fix_versions"].split()) for row in rows])
    if fix:
        text += "; version {} fixes {}".format(fix, "all" if not without_fix else
                                                "all but the {} without a fix".format(without_fix))
    else:
        text += "; no fixed version yet"
    ids = [next((a for a in row["aliases"].split() if a.startswith("CVE-")), row["id"]) for row in rows]
    text += ". " + ", ".join(ids)
    return text if len(text) <= MESSAGE_LIMIT else text[:MESSAGE_LIMIT - 3] + "..."


def sonar_report(packages, root="."):
    """SonarQube generic issue format (one rule per severity level, one issue per vulnerable package)"""
    used = sorted({package["level"] for package in packages}, key=LEVELS.index)
    rules = [dict(id="vulnerable-dependency-" + level.lower(), engineId="pip-audit",
                  name="Dependency with known vulnerabilities ({} severity)".format(level.lower()),
                  description="A pinned Python package has known vulnerabilities in the PyPI / OSV advisory database "
                              "(pip-audit); the highest severity is {}. Upgrade the package to a fixed version "
                              "or document why it is kept.".format(level.lower()),
                  cleanCodeAttribute="TRUSTWORTHY",
                  impacts=[dict(softwareQuality="SECURITY", severity=SONAR_SEVERITY[level])]) for level in used]
    issues = []
    for package in packages:
        file_name, line = pin_location(package["name"], root)
        issues.append(dict(ruleId="vulnerable-dependency-" + package["level"].lower(), effortMinutes=30,
                           primaryLocation=dict(message=package["message"], filePath=file_name,
                                                textRange=dict(startLine=line))))
    return dict(rules=rules, issues=issues)


def analyse(audit, fetch=None, workers=8):
    """rows (one per vulnerability) and packages (one per vulnerable package, worst severity first)"""
    jobs = [(dep, vuln) for dep in audit["dependencies"] for vuln in dep.get("vulns", [])]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        levels = list(pool.map(lambda job: rate(job[1], fetch), jobs))
    rows = [dict(package=dep["name"], version=dep["version"], id=vuln["id"], aliases=" ".join(vuln.get("aliases", [])),
                 severity=level, fix_versions=" ".join(vuln.get("fix_versions", [])))
            for (dep, vuln), level in zip(jobs, levels)]
    packages = []
    for dep in audit["dependencies"]:
        own = [row for row in rows if row["package"] == dep["name"]]
        if own:
            level = min((row["severity"] for row in own), key=LEVELS.index)
            packages.append(dict(name=dep["name"], version=dep["version"], level=level, rows=own,
                                 message=package_message(dep["name"], dep["version"], own)))
    packages.sort(key=lambda p: (LEVELS.index(p["level"]), -len(p["rows"]), p["name"]))
    return rows, packages


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rate the pip-audit findings and write CSV, summary and Sonar report.")
    parser.add_argument("--audit", default="Results/Security/pip_audit.json", help="pip-audit -f json output")
    parser.add_argument("--out-dir", default="Results/Security")
    parser.add_argument("--sonar", default="dependency-audit-sonar.json")
    parser.add_argument("--root", default=".", help="repository root (pinning files)")
    args = parser.parse_args(argv)
    with open(args.audit) as file:
        audit = json.load(file)
    rows, packages = analyse(audit)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "dependency_audit.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["package", "version", "id", "aliases", "severity", "fix_versions"])
        writer.writeheader()
        writer.writerows(rows)
    total = {level: sum(row["severity"] == level for row in rows) for level in LEVELS}
    by_level = ", ".join("{} {}".format(n, level.lower()) for level, n in total.items() if n)
    lines = ["{} packages scanned, {} with known vulnerabilities, {} vulnerabilities{}".format(
        len(audit["dependencies"]), len(packages), len(rows), ": " + by_level if by_level else "")]
    for package in packages:
        file_name, line = pin_location(package["name"], args.root)
        lines.append("{:<9} {}  [{}:{}]".format(package["level"], package["message"], file_name, line))
    with open(os.path.join(args.out_dir, "dependency_audit_summary.txt"), "w") as file:
        file.write("\n".join(lines) + "\n")
    with open(args.sonar, "w") as file:
        json.dump(sonar_report(packages, args.root), file, indent=1)
    print("\n".join(lines))
    return rows, packages


if __name__ == '__main__':
    main()
