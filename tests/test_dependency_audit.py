import importlib.util
import json
import os
import pytest

"""scripts/dependency_audit.py with a made-up pip-audit report and OSV records (no network)"""

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("dependency_audit", os.path.join(REPO, "scripts", "dependency_audit.py"))
da = importlib.util.module_from_spec(spec)
spec.loader.exec_module(da)

OSV = {"PYSEC-1": {"aliases": ["GHSA-aaaa"], "severity": [{"type": "CVSS_V3",
                                                           "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H"}]},
       "GHSA-aaaa": {"database_specific": {"severity": "CRITICAL"}},
       "PYSEC-2": {"severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H"}]},
       "GHSA-bbbb": {"database_specific": {"severity": "MODERATE"}},
       "OSV-3": {"severity": [{"type": "CVSS_V4", "score": "CVSS:4.0/AV:N"}]}}

AUDIT = {"dependencies": [
    {"name": "pillow", "version": "9.1.1", "vulns": [
        {"id": "PYSEC-1", "aliases": ["CVE-2022-1"], "fix_versions": ["9.2.0"]},
        {"id": "OSV-3", "aliases": [], "fix_versions": []}]},
    {"name": "requests", "version": "2.28.1", "vulns": [
        {"id": "PYSEC-2", "aliases": ["CVE-2023-2"], "fix_versions": ["2.31.0"]},
        {"id": "GHSA-bbbb", "aliases": [], "fix_versions": ["2.32.0", "3.0"]}]},
    {"name": "numpy", "version": "1.24.3", "vulns": []}]}


@pytest.mark.parametrize("vector, score", [
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H", 7.5),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N", 6.1),
    ("CVSS:3.0/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H", 7.8),
    ("CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:C/C:H/I:H/A:H", 9.9),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N", 0.0)])
def test_cvss3_score(vector, score):
    assert da.cvss3_score(vector) == score


def test_levels():
    assert [da.level_of_score(s) for s in [9.0, 7.0, 4.0, 0.1, 0]] == ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNRATED"]
    assert da.ghsa_level({"database_specific": {"severity": "MODERATE"}}) == "MEDIUM"
    assert da.ghsa_level({"database_specific": {"severity": "UNKNOWN"}}) is None


def test_rate_prefers_github_rating():
    fetch = lambda vuln_id: OSV.get(vuln_id, {})
    assert da.rate(AUDIT["dependencies"][0]["vulns"][0], fetch) == "CRITICAL"   # GHSA alias, not CVSS 7.5
    assert da.rate(AUDIT["dependencies"][1]["vulns"][0], fetch) == "HIGH"       # CVSS 3.1 only
    assert da.rate(AUDIT["dependencies"][1]["vulns"][1], fetch) == "MEDIUM"     # GHSA record itself
    assert da.rate(AUDIT["dependencies"][0]["vulns"][1], fetch) == "UNRATED"    # CVSS 4 only


def test_fix_version():
    assert da.fix_version([{"fix_versions": ["9.2.0"]}, {"fix_versions": ["10.0.1", "9.5"]}]) == "9.5"
    assert da.fix_version([{"fix_versions": []}]) is None


def test_pin_location(tmp_path):
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / "requirements-pcrl.txt").write_text("# header\nPillow==9.1.1\nscikit_learn==1.0\n")
    (tmp_path / "docker" / "legacy-py38").mkdir()
    (tmp_path / "docker" / "legacy-py38" / "environment.yml").write_text("dependencies:\n  - brotli-python=1.0.9\n")
    (tmp_path / "docker" / "Dockerfile").write_text("FROM x\nRUN pip install pip==23.1.2 setuptools==65.5.0\n")
    root = str(tmp_path)
    assert da.pin_location("pillow", root) == ("docker/requirements-pcrl.txt", 2)
    assert da.pin_location("scikit-learn", root) == ("docker/requirements-pcrl.txt", 3)
    assert da.pin_location("Brotli", root) == ("docker/legacy-py38/environment.yml", 2)
    assert da.pin_location("setuptools", root) == ("docker/Dockerfile", 2)
    assert da.pin_location("unknown", root) == da.FALLBACK


def test_main_writes_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(da, "fetch_osv", lambda vuln_id: OSV.get(vuln_id, {}))
    audit = tmp_path / "pip_audit.json"
    audit.write_text(json.dumps(AUDIT))
    sonar = tmp_path / "sonar.json"
    rows, packages = da.main(["--audit", str(audit), "--out-dir", str(tmp_path / "out"), "--sonar", str(sonar),
                              "--root", REPO])
    assert [p["name"] for p in packages] == ["pillow", "requests"]  # worst severity first, numpy has none
    assert packages[0]["message"].startswith("pillow 9.1.1: 2 known vulnerabilities (1 critical, 1 unrated); "
                                             "version 9.2.0 fixes all but the 1 without a fix. CVE-2022-1, OSV-3")
    assert "version 2.32.0 fixes all" in packages[1]["message"]
    report = json.loads(sonar.read_text())
    assert [r["id"] for r in report["rules"]] == ["vulnerable-dependency-critical", "vulnerable-dependency-high"]
    assert report["rules"][0]["impacts"] == [{"softwareQuality": "SECURITY", "severity": "BLOCKER"}]
    assert report["issues"][0]["primaryLocation"]["filePath"] == "docker/requirements-pcrl.txt"
    assert len((tmp_path / "out" / "dependency_audit.csv").read_text().splitlines()) == 5
    summary = (tmp_path / "out" / "dependency_audit_summary.txt").read_text()
    assert summary.startswith("3 packages scanned, 2 with known vulnerabilities, 4 vulnerabilities: 1 critical, 1 high, "
                              "1 medium, 1 unrated")


def test_long_message_is_cut():
    rows = [dict(severity="LOW", fix_versions="", aliases="", id="PYSEC-{:05d}".format(i)) for i in range(200)]
    assert len(da.package_message("x", "1", rows)) == da.MESSAGE_LIMIT
