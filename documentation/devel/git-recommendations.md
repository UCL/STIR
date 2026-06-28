# Developer documentation: git recommendations and rules

If you are at the start of your developer journey, or are just confused about Git, we recommend just going ahead,
and not worry about the rules below. We will help you along.

However, if you are more familiar with Git and GitHub, please follow our recommendations below.
STIR is a large project with a long history. We do care about the Git history, so help keeping it (relatively) clean.

## Pull request (PR) merges/squashes/rebases

In contrast to other projects, STIR does not enforce a "linear Git history". We generally prefer PR merges,
as it allows easier tracking of contributions and development. However, this means that we like to keep the
Git history in a PR clean.

Git allows us to modify history (!). It is a powerful and dangerous tool. Avoid it if you can!
Nevertheless, when a PR history is too messy, the maintainers might decide to

- use tools like `rebase` to clean things up
- squash-merge it (combining all commits into one).

Neither of these  is ideal for anyone who has pulled your branch already.

## Suggestions for keeping history clean

1. [install pre-commit](./git-hooks.md) and run it before every commit (you can enforce this if you like)
2. Keep your local history clean, potentially with rebasing your local commits
   - When a `git push` fails because somebody else pushed, use `git pull --rebase` and sort out conflict then,
   such that your commits are "on top" of whatever is on github. Therefore, `git pull --rebase` often to avoid
   having to handle too many conflicts.
   - Ideally don't push tiny commits, fix-ups etc, and check if your code compiles (and even runs!, and passes tests!!)
   locally before pushing.
   - Feel free to `git commit --amend` locally, as long as you didn't push.
3. Do **not** `git push -f`. If you need to, communicate with fellow developers working on the PR.

## Commit message rules

Use [well-formed commit messages](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html)
for each change (in particular with a single "subject" line
followed by an empty line and then more details).

If suitable, prefix the first line of the commit message to help understand what is being changed.
Example prefixes: `[DOC]`, `[CMAKE]`, `[PYTHON]`, `[SWIG]`. We generally don't use a prefix for C++ changes,
features, or bug fixes.

## Minimise CI resources

As an open-source project, STIR can currently use free Continuous Integration (CI) workflows. Nevertheless,
please by mindful about the resources used, as these are substantial:

- Group your commits and only push once your code compiles and tests succeed on your machine
  (ideally you have sensible commit messages at every stage)
- Use specific keywords in the *first line* of the last commit that you push to prevent CI being run:
  - `[ci skip]` skips all CI runs (e.g. when you only change documentation, or when your update isn't ready yet)
  - `[actions skip]` does not run GitHub Actions, see [here](https://github.blog/changelog/2021-02-08-github-actions-skip-pull-request-and-push-workflows-with-skip-ci/).
  Note: this can be in the main commit message.
  - `[skip appveyor]` does not run Appveyor, see [here](https://www.appveyor.com/docs/how-to/filtering-commits/#skip-directive-in-commit-message)
- During PR review, maintainers can add "suggestions", which you can directly commit via GitHub.
  However, ***batch all commits to accept suggestions*** (which you can do [via the "changes" tab on GitHub](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/incorporating-feedback-in-your-pull-request)),
  and of course, make sure that the commit message makes sense (and that you skip CI if you want to adjust afterwards locally).
