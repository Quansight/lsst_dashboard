#!/usr/bin/env bash
lsof -ti tcp:52000 | xargs kill
nohup python -c "from lsst_dashboard.gui import dashboard; dashboard.render().show(52000)"&
