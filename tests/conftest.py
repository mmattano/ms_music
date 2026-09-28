"""Shared test setup."""

import matplotlib

# Render figures off-screen: interactive backends (e.g. macOS) would open
# windows, and plt.show() would block the test run.
matplotlib.use("Agg")
