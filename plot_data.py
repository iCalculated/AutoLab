import seaborn as sns 
from matplotlib import pyplot as plt
import pandas as pd
sns.set_context("poster")
sns.axes_style()
sns.despine()
from scipy import stats

df = pd.read_csv('data.csv')

sns.lmplot(x='frame', y='x_position', data=df,
           fit_reg=True)

# get coeffs of linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(df['frame'],df['x_position'])

# use line_kws to set line label for legend
ax = sns.regplot(x="frame", y="x_position", data=df,
 line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})

# plot legend
ax.legend()

sns.despine()
plt.show()

ax.get_figure().savefig("output.png")