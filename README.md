# astro_statistics_assistant

Install dependence
```bash
conda install numpy scipy matplotlib statsmodels astropy
```

```bash
conda install -c conda-forge scikit-learn
```

## 🛠️ Development Setup for Jupyter Notebooks

To keep the repository lightweight and ensure clean code reviews, this project uses **`nbstripout`**. 

This tool acts as a "transparent filter":
- **On Commit:** It automatically strips outputs (images, tables, etc.) so they aren't saved to the Git history.
- **On Local:** Your local `.ipynb` files **remain unchanged**, keeping all your plots and results intact.

### 1. Prerequisites
Ensure you have `nbstripout` installed in your environment:
```bash
pip install nbstripout
```

### 2. One-time Configuration

After cloning the repository, you **must** run the following commands in your terminal to activate the local Git filter. This only needs to be done once per machine.

```bash
# Set up the 'clean' filter (removes outputs when you 'git add')
git config filter.nbstripout.clean "nbstripout"

# Set up the 'smudge' filter (keeps files as-is when you 'git checkout')
git config filter.nbstripout.smudge cat

```

---

*Note: If you attempt to commit without having `nbstripout` installed or configured, Git will throw an error to prevent accidental upload of heavy output data.*