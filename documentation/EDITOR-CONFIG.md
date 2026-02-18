# Configuring your editor

Your and our life will therefore be easier if you set your editor to follow certain conventions
for formatting the code. These conventions will be enforced in the near future.

Here are some pointers. Please create a PR if your editor is not covered.

## C/C++

We use [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to enforce white-space conventions for C++.

### Emacs
Install the `clang-format.el` package, e.g. from [MELPA](https://melpa.org). You will first
need to add MELPA to the list of packages as per their documentation. Then you can do
`M-x install-package RET clang-format RET`.

An example `init.el` (to put in your `.emacs.d`) is as follows

```
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

### Visual Studio Code
Install the [CLangFormat extension](https://marketplace.visualstudio.com/items?itemName=LLVMExtensions.ClangFormat) which add shortcuts to format a selection or document.

### QTCreator
See [this blog post](https://www.qt.io/blog/2019/04/17/clangformat-plugin-qt-creator-4-9) for some info.

