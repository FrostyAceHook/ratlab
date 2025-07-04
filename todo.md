- "remoce two chars from points to user" bro wtf does this MEAN lmao
- make syntax space genuinely delay required numpy import
- make 'injects':
    ```lits._injects = {
        "e": (lambda field: getattr(Single[field], "e")),
        "pi": (lambda field: getattr(Single[field], "pi")),
    }```
    - type shi
- `flip` and `flipud`/`fliplr`
- rename `rep` to `tile` and make `rep` do the numpy thing.
- `rotate` function which does brfilters np shift
- field interaction change:
    - all methods have dflts with a throw of not implemented
    - field wrapper for non-field (yeagh i know)
    - no more _need
- super specialised comparison-result matrix:
    - new "ComparisonField"
    - doesnt support `__bool__` unless single
    - is always iterable, to allow `all` and `any`
    - natural consequence of this is empty must be alled or anyed
    - remove fieldmethod LMAO
- colour more prints (generic field, rational, empty list?)
- implement _MatrixAt
- make field rep take all objects to string, so it can homogenise the display
- shortened matrix print
- irange for vector range (start stop step)
- matrix eig
- `full` function for matrix filled with one value.
- `offdiag` for off diagonal mat construction
- short and long themselves used with `with` for dflting.
- read every numpy function and grab the good ones LMAO
- make 'prog' file which handles logo, help msg, cmdline args, coloured printing,
    error msgs, whatever
- add argument to execute without syntax changes (aka just includes the default
    ratlab imports)
- rename 'syntax' to 'engine'
- add `mat` keyword where mat[] makes matrices and alternate execution method
    (-x, --bare-lists, --bare-matrices) to change what bare [] makes.
- make a good readme
- remove the concept of 'lits' and instead make rigorous promotion structure?
- sympy integration
