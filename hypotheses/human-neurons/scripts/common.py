import json, urllib.request, urllib.parse, h5py, remfile
def asset_url(ds, path):
    q=f"https://api.dandiarchive.org/api/dandisets/{ds}/versions/draft/assets/?path={urllib.parse.quote(path)}"
    a=json.load(urllib.request.urlopen(q))["results"][0]
    return f"https://api.dandiarchive.org/api/assets/{a['asset_id']}/download/"
def open_remote(ds, path):
    return h5py.File(remfile.File(asset_url(ds, path)), "r")
