---
# VIM Cheatsheet
---
---
# VIM modes
---
    Command mode – allows you to execute commands (default mode).
    Insert mode – allows you to insert/write text.
    Visual mode – visual text selector.
---
# Basic navigation and keybinds:
---
    l = move the cursor right
    h = move the cursor left
    k = cursor up
    j = cursor down
    i = enter insert mode
    $ = move to the end of a line
    yy = copy a line
    p = paste
    d = delete a line
    x = cut a character
---
# Basic commands:
---
    esc = exit insert mode and enter command mode
    :wq = write & quit
    :q = quit
    :q! = quit without saving
    / = search for word/string occurences in a document. Example, /search-word
    :vimgrep – searches through a pattern of files. More powerful than /