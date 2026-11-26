"""Repository files for outside contributors (Issue #47; pena-meta decisions/2026-11-26-outside-contributors.md).

Pass criteria: .github/CODEOWNERS exists and assigns every file to the maintainer (one rule, `* @gunjou-like`),
so that pull requests from others request the maintainer's review. GitHub's own syntax check of the file
(the codeowners/errors API) is run in the pull request that adds it.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_codeowners_names_the_maintainer_for_every_file():
    rules = [
        line.split()
        for line in (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert rules == [["*", "@gunjou-like"]]
