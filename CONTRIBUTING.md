Contributing
============

Please help us by finding problems, discussing on the mailing lists, contributing documentation,
bug fixes or even features. Below are some brief guidelines.

## Reporting a problem

Please use our [issue-tracker]: https://github.com/UCL/STIR/issues

## Submitting a patch or new feature

For contributing any code or documentation that is non-trivial, we require a
signed Contributor License Agreement, stating clearly that your
contributions are licensed appropriately. This will normally need to be signed by your
employer/university, unless you own your own copyright.
You will have to do this only once. Please check [our website for more information](http://stir.sourceforge.net/contributing/)
and contact us for more information.

Please keep a patch focused on a single issue/feature. This is important to keep our history clean,
but will also help reviewing things and therefore speed-up acceptance.

### Process

This is our recommended process. If it sounds too daunting, ask for help.

1. Create a new issue (see above). State that you will contribute a fix if you intend to do so.
2. Create a [fork](https://help.github.com/articles/fork-a-repo) on github and work from there.
3. Create a branch in your fork with a descriptive name and put your fixes there. If your fix is
simple you could do it on github by editing a file, otherwise clone your project (or add a remote
to your current git clone) and work as usual.
4. If your change is important, add it to the release notesfor the upcoming version, [see](https://github.com/UCL/STIR/blob/master/documentation/)
and even the [User's Guide](https://github.com/UCL/STIR/blob/master/documentation/STIR-UsersGuide.tex) or other documentation files.
5. Use [well-formed commit messages](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html)
for each change (in particular with a single "subject" line
followed by an empty line and then more details). If the change affects comments only, it is recommended to put `[ci skip]` in your subject line. This avoids unnecessary computation, and clogging our Travis/Appveyor queues.
6. Push the commits to your fork and submit a [pull request (PR)](https://help.github.com/articles/creating-a-pull-request)
(enable changes by project admins.) Give your pull request a descriptive name (i.e. don't call if *Fix #issuenumber*. Be prepared to add further commits to your branch after discussion.
In the description of the PR, add a statement about which Issue this applies to
using [a phrase such that github auto-closes the issue when merged to master](https://help.github.com/articles/closing-issues-using-keywords/).
7. Be prepared to add further commits to your branch after discussion.
Please by mindful about the resources used by our Continuous Integration (CI) workflows:
  - Group your commits and only push once your code compiles and tests succeed on your machine (ideally you have sensible commit messages at every stage)
  - Use specific keywords in the first line of the last commit that you push to prevent CI being run:
     - `[ci skip]` skips all CI runs (e.g. when you only change documentation, or when your update isn't ready yet)
     - `[actions skip]` does not run GitHub Actions, see [here](https://github.blog/changelog/2021-02-08-github-actions-skip-pull-request-and-push-workflows-with-skip-ci/).a
     - `[travis skip]` does not run Travis-CI, see [here](https://docs.travis-ci.com/user/customizing-the-build/#skipping-a-build).
8. After acceptance of your PR, go home with a nice warm feeling.

Suggested reading: 
https://help.github.com/articles/fork-a-repo/, https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project or https://guides.github.com/activities/forking/.

### A note on copyright dates and notices (and licenses)

(Almost) all STIR files start with a copyright and license header. Please do this for your files as well.
If you modify an existing file, you need to make sure the copyright header is up-to-date for your changes
(unless it's a trivial change).

If you copied code from somewhere, you need to preserve its copyright date/notice. If you copied non-STIR code,
you need to make sure its license is compatible with the STIR license, and indicate clearly what the license
of the copied code is (and follow its terms of course).

In addition, you might need to add yourself to [NOTICE.txt](https://github.com/UCL/STIR/blob/master/NOTICE.txt).

## Project rules

- Only one official, stable, up-to-date branch: **master**
    + Essentially "latest stable beta version with no known bugs
      since the last official release version"
    + Never knowingly add a bug to **master**
    + Exceptions to this rule might occur close to a release, where there might be a release-branch.
- Any work-in-progress commits should be in their own branches.
- GitHub assigns a unique number to each issue, c.f. the [issue-tracker].
- A pull request (PR) is an issue with an associated branch,
  c.f. [pull-requests]. Even for "internal" development, we prefer a PR for
  a branch to allow review and discussion.
- Branches and PRs are kept small (ideally one 'feature' only) and branch from **master**,
  not from another branch, unless required (e.g. for a release-branch). This allows
  commenting/improving/merging this branch/PR
  independent of other developments.
- Discussions on issues and PRs are preferably done on github. General questions and request for advice
  should go to the <stir-devel@sourceforge.net> mailing list.
- Contributions of new features should also update documentation and release notes. After version 1.0,
  this needs documentation needs to state something like "introduced after version 1.xxx".
- We prefer issues to be opened via [github][issue-tracker] due to the following reasons:
    + Ensures issues will never get lost in emails
        * Facilitates issue status tracking
    + Allows focused comments/discussion
        * Easy cross-referencing of related issues, PRs, and commits

[issue-tracker]: https://github.com/UCL/STIR/issues
[pull-requests]: https://github.com/UCL/STIR/pulls
