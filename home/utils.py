import base64
import binascii
import os
import re
import textwrap
import uuid
import yaml
from pathlib import Path

import dateutil.parser
from datetime import datetime, timedelta
from logzero import logger

BOUNDARY_START = '-----BEGIN OPENDIAMOND SCOPECOOKIE-----\n'
BOUNDARY_END = '-----END OPENDIAMOND SCOPECOOKIE-----\n'
COOKIE_VERSION = 1
BASE64_RE = '[A-Za-z0-9+/=\n]+'

def parse_cookie(data):
    """Parse a (single) scope cookie string and return a ScopeCookie
    
    Arguments:
        data {str} -- A single base64-encoded cookie

    Returns:
        [ScopeCookie] -- [description]
    """
    assert isinstance(data, str)

    # Check for boundary markers and remove them
    match = re.match(BOUNDARY_START + '(' + BASE64_RE + ')' +
                     BOUNDARY_END, data)
    if match is None:
        raise logger.error('Invalid boundary markers')
    data = match.group(1)
    # Base64-decode contents
    try:
        data = base64.b64decode(data).decode()
    except TypeError:
        raise logger.error('Invalid Base64 data')
    # Split signature, header, and body
    try:
        signature, data = data.split('\n', 1)
        header, body = data.split('\n\n', 1)
    except ValueError:
        raise logger.error('Malformed cookie')
    # Decode signature
    try:
        signature = binascii.unhexlify(signature)
    except TypeError:
        raise logger.error('Malformed signature')
    # Parse headers
    blaster = None
    for line in header.splitlines():
        k, v = line.split(':', 1)
        v = v.strip()
        if k == 'Servers':
            servers = [s.strip() for s in re.split('[;,]', v)
                       if s.strip() != '']
    # Parse body
    scopeurls = [s for s in [u.strip() for u in body.split('\n')]
                 if s != '']
    scopeurl = "/srv/diamond/INDEXES/GIDIDX"+scopeurls[0].split('/')[-1]
    print(scopeurl)
    print(servers)
    return scopeurl, servers 


def define_scope(config):
    cookie_path = Path.home() / ".hawk/NEWSCOPE"
    cookie_data = open(cookie_path).read()
    index_path, scouts = parse_cookie(cookie_data)
    config['scouts'] = scouts
    config['dataset']['index_path'] = index_path
    return config

def write_config(config, config_path):
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return 