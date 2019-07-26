# run: `when-changed gui.py "bash ./run_dashboard.sh"` in one terminal 
# run: `tail -f nohup.out` in another terminal
lsof -ti tcp:52000 | xargs kill
nohup python -c "from lsst_dashboard.gui import dashboard; dashboard.render().show(52000)"&
