"""
pandas utilities
"""

import numpy as np
import pandas as pd


def stats_agg_dataframe(
    df,
    gcol,
    ycol=None,
    functions=None,
    collapse="_",
    droplist=["mean"],
    drop_std=True,
    drop_size=True,
    conf=0.95,
    global_size="size",
    swaplevel=True,
):
    """
    Perform common stats on frame

    Parameters
    ----------

    df: DataFrame
        frame to aggregate

    gcol: list
          columns to groupby

    ycol: list
          columns to aggregate over.
          if ycol==None, use all columns less gcol

    collapse: bool or str
      if 'True': collapse Multiindex columns to regular index with '_'
      if 'False' or None: do nothing
      if str: collapse Mutiindex with character collapse

    droplist: list like
     if 'False' (or evaluates False): nothing
     otherwise, loop over droplist and drop 'collapse'+droplist[i]
     from column names. (default=['mean'])

    drop_std : bool (default True)
        if True, drop standard deviations

    drop_size : bool (default True)
        if True, drop local size


    conf : float or None:
    if not None, than

    global_size:
      if type(global_size) is str, then add a column of name
      'global_size' with size of each group

    swaplevel : bool, default=True
        if True, and no collapse, then return swap aggregotor level and name level



    Returns
    -------

    dfA: DataFrame
      Aggregated DataFrame

    """

    functions = ["mean"]

    if not drop_std or conf:
        functions.append("std")

    if not drop_size or conf or global_size is not None:
        functions.append("count")

    if ycol is None:
        ycol = list(df.columns.drop(gcol))

    # group, mean, and swap
    t = df[gcol + ycol].groupby(gcol).agg(functions).swaplevel(i=-1, j=0, axis=1)



    if conf:
        from scipy.stats import t as tdist
        SE = t["std"] / np.sqrt(t["count"]) 
        # calculate Standard error frame:
        SE = t["std"] * tdist.isf(0.5 * (1.0 - conf), t["count"] - 1)

        # add in SE label
        SE = pd.concat({"SE": SE}, axis=1)

        # add into frame
        t = pd.concat((t, SE), axis=1)

    if global_size is not None:
        name = functions[0]
        # just pick the first one
        t.loc[:, (name, global_size)] = t.loc[:, ("count", ycol[0])]

    if drop_size:
        t = t.drop("count", axis=1, level=0)

    if drop_std:
        t = t.drop("std", axis=1, level=0)

    # collapse
    if collapse:
        # swap names back to end
        t = t.swaplevel(i=0, j=-1, axis=1)
        # collapse column names
        if type(collapse) is str:
            c = collapse
        else:
            c = "_"
        # join and strip trailing separators
        t.columns = [c.join(x).rstrip(c) for x in t.columns]

        # drop '_mean'?
        if droplist:
            for d in droplist:
                t.columns = [x.replace(c + d, "") for x in t.columns]
    elif swaplevel:
        t = t.swaplevel(i=0, j=-1, axis=1)

    return t
