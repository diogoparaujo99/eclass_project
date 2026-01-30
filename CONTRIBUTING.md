# Contributing to Localization with HMMs Project

Thank you for your interest in contributing to this project! This guide will help you understand how to contribute effectively.

## How to Contribute

1. **Fork the repository** to your own GitHub account
2. **Clone your fork** locally
3. **Create a feature branch** for your changes
4. **Make your changes** and commit them with clear messages
5. **Push to your fork** and submit a pull request

## Merging Pull Requests Without Deleting Branches

When merging a pull request to the main branch, GitHub provides an option to delete the branch after merging. If you want to **keep the branch** after merging, follow these steps:

### For Repository Maintainers

#### Method 1: Uncheck the Delete Option (Recommended)

1. Navigate to the pull request you want to merge
2. Click the **"Merge pull request"** button
3. **IMPORTANT**: Before confirming, look for the checkbox that says **"Delete branch after merge"**
4. **Uncheck this box** if you want to keep the branch
5. Confirm the merge by clicking **"Confirm merge"**

The branch will remain available after the merge is complete.

#### Method 2: Configure Repository Settings

You can disable automatic branch deletion for the entire repository:

1. Go to your repository on GitHub
2. Click on **Settings** (you need admin access)
3. Navigate to **General** settings
4. Scroll down to the **Pull Requests** section
5. **Uncheck** the option: **"Automatically delete head branches"**
6. Click **Save** if needed

With this setting disabled, branches will NOT be automatically deleted after merging pull requests.

### For Pull Request Authors

If you're creating a pull request and want to ensure your branch isn't deleted:

1. Add a note in your PR description requesting that the branch not be deleted
2. Communicate with the maintainers if the branch serves ongoing purposes
3. Consider using protected branches for long-lived feature branches

### Why Keep Branches?

You might want to keep branches after merging for several reasons:

- **Future reference**: The branch name provides context about the feature
- **Continuous development**: You plan to make additional changes to the same feature
- **Documentation**: The branch serves as a milestone or reference point
- **Backup**: You want to maintain a backup of the development history

### Command Line Merge (Alternative)

If you prefer using Git command line to merge without using GitHub's interface:

```bash
# Switch to the main branch
git checkout main

# Pull the latest changes
git pull origin main

# Merge the feature branch without deleting it
git merge --no-ff feature-branch-name

# Push the merge to remote
git push origin main

# The feature branch remains available locally and remotely
```

The feature branch will still exist both locally and on GitHub after this merge.

## Reporting Issues

If you find any problems or bugs, please open an issue on GitHub with:

- A clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment details (Python version, OS, etc.)

## Pull Request Guidelines

- Write clear commit messages
- Include tests for new functionality
- Update documentation as needed
- Follow the existing code style
- Reference any related issues

## Questions?

If you have questions about contributing, feel free to open an issue or contact the maintainers.

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project.
