# CIFAR-10 Projects

This workspace contains two related CIFAR-10 implementations:

- `desktop-client/`: a WPF desktop client for classifying images through a local API
- `python-backend/`: the Python training, inference, and Flask API project

## Recommended Entry Points

- `python-backend/README.md` for the model, training, and inference workflow
- `desktop-client/MainWindow.xaml` for the desktop UI

## Clean Project Layout

Generated build outputs are intentionally excluded from the workspace view so the source folders stay easy to read:

- `.vs/`
- `bin/`
- `obj/`
- `__pycache__/`
- notebook checkpoints

If you plan to publish this as a GitHub repository, keep the source folders and the README files, and leave the generated outputs out of version control.
