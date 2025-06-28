- "remoce two chars from points to user" bro wtf does this MEAN lmao
- numpy backing + remove generic field
- add `reshape`
- exposed attrs as actual class attributes + docs.
- super specialised bool matrix:
    - doesnt support `__bool__` unless single
    - is always iterable, to allow `all` and `any`
    - natural consequence of this is empty must be alled or anyed
- colour more prints (generic field, rational, empty list?)
- figure out indexing via mask
- make field rep take all objects to string, so it can homogenise the display
- `rotate` function which does brfilters np shift
- shortened matrix print
- matrix eig
- maybe restructure matrix to only be "instantiated" when required and tack a
    "callable" onto each one. this would allow things like `x.summ` and
    `x.summ(0)`(meaning `x.summ_along(0)`) to work. however, its like kinda a
    terrible idea.
- sympy integration
