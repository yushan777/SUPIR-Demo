source venv/bin/activate

# To enable clipboard paste from client PC, use an SSH tunnel instead of --share.
# The tunnel makes the browser see the connection as localhost (a secure context),
# which allows the clipboard API to work without routing traffic over the internet.
# Run this on the client PC (change username and ip as necessary):
#   ssh -L 3000:localhost:3000 -p 1991 user@192.168.0.50
# Then access http://localhost:3000 in the browser.

# python3 run_supir_gradio.py
python3 run_supir_gradio.py --listen --port 3000
