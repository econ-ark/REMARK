@echo off
rem = """
jupyter contrib nbextension install --user
jupyter nbextension enable codefolding/main
python -m cite2c.install
python -x "%~f0" %*
exit /b %errorlevel%
:: End of batch file commands
"""

from notebook.services.config.manager import ConfigManager
cm = ConfigManager()
cm.update('cite2c', {'zotero':{"user_id": "5043554","username": "econ-ark","access_token": "XZpH9NsoAZmDMmjLKiy8xMXX"}})
