import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Example dataset
data = {'Group': ['A']*5 + ['B']*5 + ['C']*5,
        'Values': [25, 30, 35, 40, 45, 20, 22, 25, 28, 30, 15, 18, 20, 25, 30]}

df = pd.DataFrame(data)

# Fit the ANOVA model
model = ols('Values ~ Group', data=df).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

# Print ANOVA table
print("ANOVA Table:")
print(anova_table)

# Perform Tukey's HSD test for post-hoc analysis
posthoc = pairwise_tukeyhsd(df['Values'], df['Group'], alpha=0.05)

# Print the summary of the post-hoc test
print("\nTukey's HSD Post-Hoc Test:")
print(posthoc)
