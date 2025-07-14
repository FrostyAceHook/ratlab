- "remoce two chars from points to user" bro wtf does this MEAN lmao
- `zeros`/`ones` do NOT default to square
- `flip` and `flipud`/`fliplr`
- rename `rep` to `tile` and make `rep` do the numpy thing.
- `rotate` function which does brfilters np shift
- `matstack`:
def matstack(*xs, field=None):
    """
    Concatenates the given matrices along the third axis (index 2).
    """
    return stack(2, *xs, field=field)
- `max`/`min` do take-min/max when given multiple (stack then max_along type shi)
- add hyperbolic trig
- matrix eig
- `full` function for matrix filled with one value.
- `offdiag` for off diagonal mat construction
- `angle` for complex arg.
- `diag` takes multiple args
- `isinf`/`isnan`/`isfinite`
- `isreal` and `isimag` (along w rigorous treatment of -0.0i)
- formula descriptions for means
- read every numpy function and grab the good ones LMAO
- colour more prints (generic field, rational, empty list?)
- implement _MatrixAt
- linspace/logspace/arange as field mat methods.
- make field rep take all objects to string, so it can homogenise the display
- optimise Field.rep
- shortened matrix print
- short and long themselves used with `with` for dflting.
- `ans` should be able to access attrs of it but never be set?
- make 'prog' nicely handle errors
- make a good readme
- make 'prog' parser good lmao
- parse all filenames at the start of execution (but obv still check before each)
- make a no-colour arg
- iroot/ilog
- fix exception coloured printing when the exception message contains newlines.
    also like if the error spans lines what happens?
- vectorise all 2d operations for ndim>2 to drop first two axes?
- remove the concept of 'lits' and instead make rigorous promotion structure
- add fieldmethod again and integrate it with field promotion type shit
- sympy integration
- add `ef` and `pif` for floating e and pi, and make `e` and `pi` symbolic
