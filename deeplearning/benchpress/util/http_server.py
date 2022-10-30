# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import portpicker
import queue
import multiprocessing
import waitress
import subprocess
import json
import typing
import requests
import flask

from absl import flags

from deeplearning.benchpress.util import logging as l
from deeplearning.benchpress.util import environment

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "use_http_server",
  False,
  "Select to use http server in the app. If you set to True, the app will know how to use it with respect to the requested task."
)

flags.DEFINE_integer(
  "http_port",
  40822,
  "Define port this current server listens to."
)

flags.DEFINE_string(
  "http_server_ip_address",
  "cc1.inf.ed.ac.uk",
  "Set the target IP address of the host http server."
)

flags.DEFINE_list(
  "http_server_peers",
  None,
  "Set comma-separated http address to load balance on secondary nodes."
)

flags.DEFINE_string(
  "host_address",
  "localhost",
  "Specify address where http server will be set."
)

app = flask.Flask(__name__)

class FlaskHandler(object):
  def __init__(self):
    self.read_queue   = None
    self.write_queues = None
    self.reject_queue = None
    self.backlog      = None
    return

  def set_params(self, read_queue, write_queues, reject_queues, manager, work_flag):
    self.read_queue    = read_queue
    self.write_queues  = write_queues
    self.work_flag     = work_flag
    self.reject_queues = reject_queues
    self.manager       = manager
    self.backlog       = []
    return

handler = FlaskHandler()

@app.route('/write_message', methods=['PUT'])
def write_message(): # Expects serialized json file, one list of dictionaries..
  """
  This function receives new kernels that need to be computed.

  Collect a json file with data and send to computation..

  Example command:
    curl -X PUT http://localhost:PORT/write_message \
         --header "Content-Type: application/json" \
         -d @/path/to/json/file.json
  """
  source = flask.request.headers.get("Server-Name")
  if source is None:
    return "Source address not provided.", 404
  if source not in handler.write_queues:
    handler.write_queues[source] = handler.manager.list()
  if source not in handler.reject_queues:
    handler.reject_queues[source] = handler.manager.list()
  data = flask.request.json
  if not isinstance(data, list):
    return "ERROR: JSON Input has to be a list of dictionaries. One for each entry.\n", 400
  for entry in data:
    handler.read_queue.put([source, entry])
  return 'OK\n', 200

@app.route('/read_message', methods = ['GET'])
def read_message() -> bytes:
  """
  Publish all the predicted results of the write_queue.
  Before flushing the write_queue, save them into the backlog.

  Example command:
    curl -X GET http://localhost:PORT/read_message
  """
  source = flask.request.headers.get("Server-Name")
  ret = [r for r in handler.write_queues[source]]
  handler.write_queues[source] = handler.manager.list()
  handler.backlog += [[source, r] for r in ret]
  return bytes(json.dumps(ret), encoding="utf-8"), 200

@app.route('/read_rejects', methods = ['GET'])
def read_rejects() -> bytes:
  """
  Publish all the predicted results of the write_queue.
  Before flushing the write_queue, save them into the backlog.

  Example command:
    curl -X GET http://localhost:PORT/read_rejects
  """
  source = flask.request.headers.get("Server-Name")
  ret = [r for r in handler.reject_queues[source]]
  return bytes(json.dumps(ret), encoding="utf-8"), 200

@app.route('/read_reject_labels', methods = ['GET'])
def read_reject_labels() -> bytes:
  """
  Get labels of rejected OpenCL kernels.

  Example command:
    curl -X GET http://localhost:PORT/read_reject_labels
  """
  labels = {}
  source = flask.request.headers.get("Server-Name")
  if source is None:
    return "Server-Name is undefined", 404
  ret = [r for r in handler.reject_queues[source]]
  for c in ret:
    if c['runtime_features']['label'] not in labels:
      labels[c['runtime_features']['label']] = 1
    else:
      labels[c['runtime_features']['label']] += 1
  return bytes(json.dumps(labels), encoding="utf-8"), 200

@app.route('/read_queue_size', methods = ['GET'])
def read_queue_size() -> bytes:
  """
  Read size of pending workload in read_queue.
  """
  return handler.read_queue.qsize(), 200

@app.route('/get_backlog', methods = ['GET'])
def get_backlog() -> bytes:
  """
  In case a client side error has occured, proactively I have stored
  the whole backlog in memory. To retrieve it, call this method.

  Example command:
    curl -X GET http://localhost:PORT/get_backlog
  """
  return bytes(json.dumps(handler.backlog), encoding = "utf-8"), 200

@app.route('/status', methods = ['GET'])
def status():
  """
  Read the workload status of the http server.
  """
  source = flask.request.headers.get("Server-Name")
  if source is None:
    return "Server-Name is undefined", 404
  status = {
    'read_queue'        : 'EMPTY' if handler.read_queue.empty() else 'NOT_EMPTY',
    'write_queue'       : 'EMPTY' if len(handler.write_queues[source]) == 0 else 'NOT_EMPTY',
    'reject_queue'      : 'EMPTY' if len(handler.reject_queues[source]) == 0 else 'NOT_EMPTY',
    'work_flag'         : 'WORKING' if handler.work_flag.value else 'IDLE',
    'read_queue_size'   : handler.read_queue.qsize(),
    'write_queue_size'  : len(handler.write_queues[source]),
    'reject_queue_size' : len(handler.reject_queues[source]),
  }

  if status['read_queue'] == 'EMPTY' and status['write_queue'] == 'EMPTY':
    return bytes(json.dumps(status), encoding = 'utf-8'), 200 + (100 if handler.work_flag.value else 0)
  elif status['read_queue'] == 'EMPTY' and status['write_queue'] == 'NOT_EMPTY':
    return bytes(json.dumps(status), encoding = 'utf-8'), 201 + (100 if handler.work_flag.value else 0)
  elif status['read_queue'] == 'NOT_EMPTY' and status['write_queue'] == 'EMPTY':
    return bytes(json.dumps(status), encoding = 'utf-8'), 202 + (100 if handler.work_flag.value else 0)
  elif status['read_queue'] == 'NOT_EMPTY' and status['write_queue'] == 'NOT_EMPTY':
    return bytes(json.dumps(status), encoding = 'utf-8'), 203 + (100 if handler.work_flag.value else 0)

@app.route('/', methods = ['GET', 'POST', 'PUT'])
def index():
  """
  In case a client side error has occured, proactively I have stored
  the whole backlog in memory. To retrieve it, call this method.

  Example command:
    curl -X GET http://localhost:PORT/get_backlog
  """
  multi_status = {
    'read_queue'      : 'EMPTY' if handler.read_queue.empty() else 'NOT_EMPTY',
    'read_queue_size' : handler.read_queue.qsize(),
    'work_flag'       : 'WORKING' if handler.work_flag.value else 'IDLE',
  }
  it = set(handler.write_queues.keys())
  it.update(set(handler.reject_queues.keys()))
  multi_status['out_servers'] = {}
  for hn in it:
    status = {
      'write_queue'       : 'EMPTY' if hn in handler.write_queues and len(handler.write_queues[hn]) == 0 else 'NOT_EMPTY',
      'reject_queue'      : 'EMPTY' if hn in handler.reject_queues and len(handler.reject_queues[hn]) == 0 else 'NOT_EMPTY',
      'write_queue_size'  : len(handler.write_queues[hn]) if hn in handler.write_queues else 0,
      'reject_queue_size' : len(handler.reject_queues[hn]) if hn in handler.reject_queues else 0,
    }
    multi_status['out_servers'][hn] = status
    return flask.render_template("index.html", data = multi_status)

def http_serve(read_queue    : multiprocessing.Queue,
               write_queues  : multiprocessing.Manager.dict,
               reject_queues : multiprocessing.Manager.dict,
               work_flag     : multiprocessing.Value,
               peers         : multiprocessing.Manager.list,
               manager       : multiprocessing.Manager,
               ) -> None:
  """
  Run http server for read and write workload queues.
  """
  try:
    port = FLAGS.http_port
    if port is None:
      port = portpicker.pick_unused_port()
    handler.set_params(read_queue, write_queues, reject_queues, manager, work_flag)
    hostname = subprocess.check_output(
      ["hostname", "-i"],
      stderr = subprocess.STDOUT,
    ).decode("utf-8").replace("\n", "").split(' ')
    if len(hostname) == 2:
      ips = "ipv4: {}, ipv6: {}".format(hostname[1], hostname[0])
    else:
      ips = "ipv4: {}".format(hostname[0])
    l.logger().warn("Server Public IP: {}".format(ips))
    waitress.serve(app, host = FLAGS.host_address, port = port)
  except KeyboardInterrupt:
    return
  except Exception as e:
    raise e
  return

def client_status_request() -> typing.Tuple[typing.Dict, int]:
  """
  Get status of http server.
  """
  try:
    if FLAGS.http_port == -1:
      r = requests.get("{}/status".format(FLAGS.http_server_ip_address), headers = {"Server-Name": environment.HOSTNAME})
    else:
      r = requests.get("http://{}:{}/status".format(FLAGS.http_server_ip_address, FLAGS.http_port), headers = {"Server-Name": environment.HOSTNAME})
  except Exception as e:
    l.logger().error("GET status Request at {}:{} has failed.".format(FLAGS.http_server_ip_address, FLAGS.http_port))
    raise e
  return r.json(), r.status_code

def client_get_request() -> typing.List[typing.Dict]:
  """
  Helper function to perform get request at /read_message of http target host.
  """
  try:
    if FLAGS.http_port == -1:
      r = requests.get("{}/read_message".format(FLAGS.http_server_ip_address), headers = {"Server-Name": environment.HOSTNAME})
    else:
      r = requests.get("http://{}:{}/read_message".format(FLAGS.http_server_ip_address, FLAGS.http_port), headers = {"Server-Name": environment.HOSTNAME})
  except Exception as e:
    l.logger().error("GET Request at {}:{} has failed.".format(FLAGS.http_server_ip_address, FLAGS.http_port))
    raise e
  if r.status_code == 200:
    return r.json()
  else:
    l.logger().error("Error code {} in read_message request.".format(r.status_code))
  return None

def client_put_request(msg: typing.List[typing.Dict]) -> None:
  """
  Helper function to perform put at /write_message of http target host.
  """
  try:
    if FLAGS.http_port == -1:
      r = requests.put("{}/write_message".format(FLAGS.http_server_ip_address), data = json.dumps(msg), headers = {"Content-Type": "application/json", "Server-Name": environment.HOSTNAME})
    else:
      r = requests.put("http://{}:{}/write_message".format(FLAGS.http_server_ip_address, FLAGS.http_port), data = json.dumps(msg), headers = {"Content-Type": "application/json", "Server-Name": environment.HOSTNAME})
  except Exception as e:
    l.logger().error("PUT Request at {}:{} has failed.".format(FLAGS.http_server_ip_address, FLAGS.http_port))
    raise e
  if r.status_code != 200:
    l.logger().error("Error code {} in write_message request.".format(r.status_code))
  return

def start_server_process() -> typing.Tuple[multiprocessing.Process, multiprocessing.Value, multiprocessing.Queue, typing.Dict, typing.Dict]:
  """
  This is an easy wrapper to start server from parent routine.
  Starts a new process or thread and returns all the multiprocessing
  elements needed to control the server.
  """
  m = multiprocessing.Manager()
  rq, wqs, rjqs, peers = multiprocessing.Queue(), m.dict(), m.dict(), m.list()
  wf = multiprocessing.Value('i', False)
  p = multiprocessing.Process(
    target = http_serve,
    kwargs = {
      'read_queue'    : rq,
      'write_queues'  : wqs,
      'reject_queues' : rjqs,
      'work_flag'     : wf,
      'peers'         : peers
      'manager'       : m,
    }
  )
  p.daemon = True
  p.start()
  return p, wf, rq, wqs, rjqs
