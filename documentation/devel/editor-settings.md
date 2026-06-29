# Developer documentation: editor settings

White-spaces and indentation with multiple developers are a pain. Please adhere to
our white-space policy, which we enforce via [pre-commit](https://pre-commit.com/), including
running of [clang-format](https://clang.llvm.org/docs/ClangFormat.html).
See [git-hooks.md](git-hooks.md) for more information.

Developer experience will be best if you set your editor/IDE to use the same settings,
ideally including versions of the formatters (`clang-format` for C++ in particular),
as set in `.pre-commit-config.yaml`.
However, even if you have the version incorrect, in many cases it will help (and minimise
changes introduced by our `pre-commit` hooks.)

Here are some pointers. Feel free to create a PR if your editor is not covered.

## C/C++

We use [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to enforce white-space conventions for C++.

### Emacs

Install the `clang-format.el` package, e.g. from [MELPA](https://melpa.org). You will first
need to add MELPA to the list of packages as per their documentation. Then you can do
`M-x install-package RET clang-format RET`.

An example `init.el` (to put in your `.emacs.d`) is as follows

```lisp
(setq-default indent-tabs-mode nil)
;; save history
(savehist-mode 1)

;; add MELPA package archive
(require 'package)
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/") t)
;; Comment/uncomment this line to enable MELPA Stable if desired.  See `package-archive-priorities`
;; and `package-pinned-packages`. Most users will not need or want to do this.
;;(add-to-list 'package-archives '("melpa-stable" . "https://stable.melpa.org/packages/") t)
(package-initialize)

;; these lines were added automatically after M-x install-package RET clang-format RET
(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(package-selected-packages (quote (clang-format))))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 )


;; Manual Clang stuff if not use MELPA

;; (require 'clang-format)
;; (setq clang-format-style "file")

;; We can now use M-x clang-format-buffer and M-x clang-format-region.
;; For convenience, bind the latter to a key
(global-set-key [C-M-tab] 'clang-format-region)

;; add .inl and .txx file to extensions for automatic c++-mode
(add-to-list 'auto-mode-alist '("\\.inl\\'" . c++-mode))
(add-to-list 'auto-mode-alist '("\\.txx\\'" . c++-mode))
```

### Visual Studio 2019 (or later)

VS automatically finds our `.clang-format` in the source tree and adds 2 commands to the `Tools`
menu to format a selection or document (with shortcuts).
See the [Visual Studio documentation](https://learn.microsoft.com/en-us/visualstudio/ide/reference/options-text-editor-c-cpp-formatting?view=visualstudio).

### Visual Studio Code

See the [VS Code documentation](https://code.visualstudio.com/docs/cpp/cpp-ide#_code-formatting).

### QTCreator

See [this blog post](https://www.qt.io/blog/2019/04/17/clangformat-plugin-qt-creator-4-9) for some info.

### Eclipse

This extension might help: https://marketplace.eclipse.org/free-tagging/clang-format.
