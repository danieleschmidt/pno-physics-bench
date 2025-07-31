#!/usr/bin/env python3
"""Release management script for PNO Physics Bench."""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
import json
from datetime import datetime


def run_command(cmd: str, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd, shell=True, capture_output=capture_output, text=True
    )
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        if capture_output:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        sys.exit(1)
    return result


def get_current_version() -> str:
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")
    
    content = pyproject_path.read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError("Version not found in pyproject.toml")
    
    return match.group(1)


def update_version(new_version: str) -> None:
    """Update version in pyproject.toml and __init__.py."""
    # Update pyproject.toml
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    content = re.sub(
        r'version\s*=\s*"[^"]+"',
        f'version = "{new_version}"',
        content
    )
    pyproject_path.write_text(content)
    print(f"Updated version in pyproject.toml to {new_version}")
    
    # Update __init__.py
    init_path = Path("src/pno_physics_bench/__init__.py")
    if init_path.exists():
        content = init_path.read_text()
        content = re.sub(
            r'__version__\s*=\s*"[^"]+"',
            f'__version__ = "{new_version}"',
            content
        )
        init_path.write_text(content)
        print(f"Updated version in __init__.py to {new_version}")


def validate_version(version: str) -> bool:
    """Validate version format (semantic versioning)."""
    pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-(alpha|beta|rc)\.(\d+))?(?:\.dev(\d+))?$'
    return bool(re.match(pattern, version))


def parse_version(version: str) -> Tuple[int, int, int, Optional[str], Optional[int]]:
    """Parse version string into components."""
    pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-(alpha|beta|rc)\.(\d+))?(?:\.dev(\d+))?$'
    match = re.match(pattern, version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")
    
    major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
    prerelease = match.group(4)
    prerelease_num = int(match.group(5)) if match.group(5) else None
    
    return major, minor, patch, prerelease, prerelease_num


def bump_version(current: str, bump_type: str) -> str:
    """Bump version based on type."""
    major, minor, patch, prerelease, prerelease_num = parse_version(current)
    
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    elif bump_type == "alpha":
        if prerelease == "alpha":
            return f"{major}.{minor}.{patch}-alpha.{prerelease_num + 1}"
        else:
            return f"{major}.{minor}.{patch + 1}-alpha.1"
    elif bump_type == "beta":
        if prerelease == "beta":
            return f"{major}.{minor}.{patch}-beta.{prerelease_num + 1}"
        else:
            return f"{major}.{minor}.{patch + 1}-beta.1"
    elif bump_type == "rc":
        if prerelease == "rc":
            return f"{major}.{minor}.{patch}-rc.{prerelease_num + 1}"
        else:
            return f"{major}.{minor}.{patch + 1}-rc.1"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def update_changelog(version: str) -> None:
    """Update CHANGELOG.md with release information."""
    changelog_path = Path("CHANGELOG.md")
    if not changelog_path.exists():
        print("CHANGELOG.md not found, skipping update")
        return
    
    content = changelog_path.read_text()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Replace [Unreleased] with version and date
    content = content.replace(
        "## [Unreleased]",
        f"## [Unreleased]\n\n## [{version}] - {today}"
    )
    
    changelog_path.write_text(content)
    print(f"Updated CHANGELOG.md for version {version}")


def check_working_directory() -> None:
    """Check if working directory is clean."""
    result = run_command("git status --porcelain")
    if result.stdout.strip():
        print("Working directory is not clean. Please commit or stash changes.")
        sys.exit(1)


def check_branch() -> None:
    """Check if on main/master branch."""
    result = run_command("git branch --show-current")
    branch = result.stdout.strip()
    if branch not in ["main", "master"]:
        print(f"Not on main/master branch (current: {branch})")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            sys.exit(1)


def run_tests() -> None:
    """Run test suite."""
    print("Running test suite...")
    run_command("python -m pytest tests/ -v", capture_output=False)


def create_tag(version: str, message: str = None) -> None:
    """Create git tag."""
    tag_name = f"v{version}"
    if message is None:
        message = f"Release {tag_name}"
    
    run_command(f'git tag -a {tag_name} -m "{message}"')
    print(f"Created tag {tag_name}")


def push_changes(tag: str = None) -> None:
    """Push changes and optionally tag."""
    if tag:
        run_command("git push origin main")
        run_command(f"git push origin {tag}")
        print(f"Pushed changes and tag {tag}")
    else:
        run_command("git push origin main")
        print("Pushed changes")


def create_release_notes(version: str) -> str:
    """Create release notes from changelog."""
    changelog_path = Path("CHANGELOG.md")
    if not changelog_path.exists():
        return f"Release {version}"
    
    content = changelog_path.read_text()
    
    # Extract section for this version
    pattern = rf"## \[{re.escape(version)}\].*?\n(.*?)(?=\n## \[|$)"
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return f"Release {version}"


def main():
    parser = argparse.ArgumentParser(description="Release management for PNO Physics Bench")
    parser.add_argument(
        "action",
        choices=["bump", "release", "check"],
        help="Action to perform"
    )
    parser.add_argument(
        "--type",
        choices=["major", "minor", "patch", "alpha", "beta", "rc"],
        help="Version bump type"
    )
    parser.add_argument(
        "--version",
        help="Specific version to release"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force release even with uncommitted changes"
    )
    
    args = parser.parse_args()
    
    if args.action == "check":
        # Check release readiness
        print("Checking release readiness...")
        
        try:
            current_version = get_current_version()
            print(f"Current version: {current_version}")
            
            if not args.force:
                check_working_directory()
                check_branch()
            
            if not args.skip_tests:
                run_tests()
            
            print("✅ Ready for release!")
            
        except Exception as e:
            print(f"❌ Not ready for release: {e}")
            sys.exit(1)
    
    elif args.action == "bump":
        if not args.type and not args.version:
            print("Must specify --type or --version for bump")
            sys.exit(1)
        
        current_version = get_current_version()
        print(f"Current version: {current_version}")
        
        if args.version:
            new_version = args.version
        else:
            new_version = bump_version(current_version, args.type)
        
        if not validate_version(new_version):
            print(f"Invalid version format: {new_version}")
            sys.exit(1)
        
        print(f"New version: {new_version}")
        
        if args.dry_run:
            print("Dry run - no changes made")
        else:
            update_version(new_version)
            
            # Commit version bump
            run_command("git add pyproject.toml src/pno_physics_bench/__init__.py")
            run_command(f'git commit -m "Bump version to {new_version}"')
            
            print(f"Version bumped to {new_version}")
    
    elif args.action == "release":
        if not args.force:
            check_working_directory()
            check_branch()
        
        current_version = get_current_version()
        print(f"Creating release for version: {current_version}")
        
        if not args.skip_tests:
            run_tests()
        
        if args.dry_run:
            print("Dry run - no changes made")
            print(f"Would create tag v{current_version}")
            print("Would update CHANGELOG.md")
            print("Would push to origin")
        else:
            # Update changelog
            update_changelog(current_version)
            
            # Commit changelog update
            run_command("git add CHANGELOG.md")
            run_command(f'git commit -m "Update CHANGELOG for {current_version}"')
            
            # Create tag
            create_tag(current_version)
            
            # Push changes and tag
            push_changes(f"v{current_version}")
            
            print(f"✅ Release {current_version} created!")
            print("GitHub Actions will handle the rest of the release process.")


if __name__ == "__main__":
    main()