"""Build and validate an installed wheel outside the checkout.

Use --offline to borrow installed dependency packages; CI uses a clean environment.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='valen-wheel-') as temporary:
        temporary = Path(temporary)
        subprocess.run([sys.executable,'-m','pip','wheel','--no-build-isolation','--no-deps','--wheel-dir',str(temporary),str(root)],check=True)
        environment = temporary/'environment'
        venv.EnvBuilder(with_pip=True, system_site_packages=options.offline).create(environment)
        python = environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        wheel = next(temporary.glob('*.whl'))
        env = {key:value for key,value in os.environ.items() if key != 'PYTHONPATH'}
        subprocess.run([str(python),'-m','pip','install',*(['--no-deps'] if options.offline else []),str(wheel)],check=True,env=env,cwd=temporary)
        script = '''
import asyncio, sys
from pathlib import Path
import httpx, valen_sdk
from valen_sdk import Valen, AsyncValen, Noul, ValenResponseError
assert 'site-packages' in str(Path(valen_sdk.__file__).resolve())
assert not {'torch','transformers','onnxruntime','numpy'} & sys.modules.keys()
question = {'flag':Noul(instructions='Flag?')}
def response(request):
    return httpx.Response(200,json={'model':'installed','answers':{'flag':{'type':'noul','noul':.8}},'usage':{}})
with Valen.remote('https://example.test/v1',transport=httpx.MockTransport(response)) as client:
    assert client.system_one('x',question).answers['flag'].noul == .8
async def run():
    async with AsyncValen('https://example.test/v1',transport=httpx.MockTransport(response)) as client:
        assert (await client.system_one('x',question)).answers['flag'].noul == .8
asyncio.run(run())
print('Installed wheel: sync/async HTTP and inference-free import checks passed.')
'''
        subprocess.run([str(python),'-c',script],check=True,env=env,cwd=temporary)

if __name__=='__main__': main()
