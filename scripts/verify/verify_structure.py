from pathlib import Path


required = [
    'config/config.yaml', 'config/config_loader.py',
    'utils/logger.py', 'utils/seed.py', 'utils/__init__.py',
    'requirements.txt', 'setup.py', '.env.example', '.gitignore',
    'README.md', 'report/technical_report.md',
    'data/__init__.py', 'models/__init__.py',
    'inference/__init__.py', 'training/__init__.py',
    'evaluation/__init__.py', 'api/__init__.py',
    'tests/__init__.py',
]


missing = [f for f in required if not Path(f).exists()]
if missing:
    print('MISSING FILES:')
    for f in missing: print(f'  - {f}')
else: print('Section 1 complete. All files present. Proceed to Section 2.')
