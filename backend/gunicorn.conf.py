# Gunicorn configuration file for AI Citizen Support Assistant API

import os

bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")
backlog = 2048

workers = int(os.getenv("GUNICORN_WORKERS", 4))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
threads = 1

timeout = 120
keepalive = 5
graceful_timeout = 30

max_requests = 1000
max_requests_jitter = 50

daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

proc_name = "aria-api"

# Hooks
def on_starting(server):
    """Called just before the master process is initialized."""
    pass

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    pass

def when_ready(server):
    """Called just after the server is started."""
    pass

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    pass

def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    pass

def worker_int(worker):
    """Called when a worker receives SIGINT or SIGQUIT."""
    pass

def worker_abort(worker):
    """Called when a worker receives SIGABRT."""
    pass

def pre_exec(server):
    """Called just before a new master process is forked."""
    pass

def child_exit(server, worker):
    """Called in the master process after a worker has exited."""
    pass

def worker_exit(server, worker):
    """Called in the worker process just after a worker has exited."""
    pass

def nworkers_changed(server, new_value, old_value):
    """Called when the number of workers is changed."""
    pass

def on_exit(server):
    """Called just before the master process exits."""
    pass
