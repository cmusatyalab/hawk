python setup.py install 
systemctl restart hawk
systemctl is-active hawk >/dev/null 2>&1 && echo "hawk is active" || echo "hawk NOT active"
