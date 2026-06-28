# Developer documentation: editor settings

White-spaces and indentation with multiple developers are a pain. Please adhere to
our white-space policy, which we enforce via [pre-commit](https://pre-commit.com/), including
running of [clang-format](https://clang.llvm.org/docs/ClangFormat.html).
See [git-hooks.md](git-hooks.md) for more information.

Developer experience will be best if you set your editor/IDE to use the same settings,
including versions of the formatters (`clang-format` for C++ in particular).
Unfortunately, this isn't very easy to get exactly right as `clang-format` output is version specific.
However, in many cases it will help (and minimise changes introduced by our `pre-commit` hooks.)

Editor documentation for C++ formatting:
- [VS Code documentation](https://code.visualstudio.com/docs/cpp/cpp-ide#_code-formatting)
- [Eclipse](https://marketplace.eclipse.org/free-tagging/clang-format)
- [Visual Studio](https://learn.microsoft.com/en-us/visualstudio/ide/reference/options-text-editor-c-cpp-formatting?view=visualstudio)
- Emacs: get [clang-format.el](https://github.com/sonatard/clang-format) from MELPA, configure it to use our `.clang-format`.