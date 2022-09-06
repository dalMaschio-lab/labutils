import json, os, requests, vedo
import numpy as np


class BrainRegion(object):
    def __init__(self, name, path, k, parent=None, children=None, color=None, broken_stl=True, flip_normal=False):
        self.name = name
        self.path = path
        self.parent = parent
        self.sub_regions = children
        self._mesh = None
        self.broken_stl = broken_stl
        self.flip_normal = flip_normal
        self.color = color
        self.idx = k
    
    @property
    def mesh(self):
        if self.broken_stl:
            return None
        elif self._mesh is None:
            self._mesh = vedo.io.load(os.path.join(self.path, 'stl.stl'))
            self._mesh.color(self.color, alpha=.3)
            if self.flip_normal:
                self._mesh.reverse(normals=True)
        else:
            pass
        return self._mesh

    def is_inside(self, points):
        return self.mesh.insidePoints(points, returnIds=True)


class MPIN_Atlas:
    json_region ='load_regions_data.json'
    def __init__(self, path):
        self.live_template = os.path.join(path, 'MPIN-Atlas__Reference_brains__Live__HuCH2BGCaMP.nrrd')
        self.std_template = os.path.join(path, 'MPIN-Atlas__Reference_brains__Fixed__HuC.nrrd')
        self.antsoptpoints = {
            'this2std': [
                "-t", f"[{path}/live2fixed_0GenericAffine.mat,1]",
                "-t", f"[{path}/live2fixed_1InverseWarp.nii.gz]",
            ],
            'std2this': [
                "-t", f"[{path}/live2fixed_1Warp.nii.gz]",
                "-t", f"[{path}/live2fixed_0GenericAffine.mat]",
            ]
        }
        with open(os.path.join(path, "url")) as fd:
            remote_url = fd.readline().strip()
        if not os.path.exists(os.path.join(path, self.json_region)):
            response = requests.get(remote_url + "/api/load_regions_data")
            response.close()
            with open(os.path.join(path, self.json_region), 'w') as fd:
                fd.write(response.text)
        self.path = path
        with open(os.path.join(path, self.json_region)) as fd:
            reg = json.load(fd)["data"]
        self.hierarchy = self._open_h(reg['regions_hierarchy'])
        self.regions = {
            0: BrainRegion("outside", None, 0, broken_stl=True, color="#000000"),
            }
        reg['regions_dictionary']['1'] = {
            'name': 'brain',
            "parent": None,
            'is_container': False,
            'sub_regions_ids': [i[0] for i in self.hierarchy],
            'downloads': {'stl': '/media/Brains/Outline/Outline_new.stl'},
            'visualization_data': {'color': "#a0a0a0"}
        }
        self.hierarchy = [(1, self.hierarchy)]
        for k, i in reg['regions_dictionary'].items():
            if not i['is_container']:
                r_path = os.path.join(path, "regions", i['downloads']['stl'].split('/')[-2])
                os.makedirs(r_path, exist_ok=True)
                broken_stl = True
                if not os.path.exists(os.path.join(r_path, "stl.stl")):
                    response = requests.get(remote_url + i['downloads']['stl'])
                    if response.status_code == 200:
                        broken_stl = False
                        with open(os.path.join(r_path, "stl.stl"), 'w') as fd:
                            fd.write(response.text)
                    else:
                        print(response.status_code, i['name'])
                    response.close()
                else:
                    broken_stl = False

                self.regions[int(k)] = BrainRegion(
                    i['name'], r_path, int(k),
                    parent=i['parent'] if i['parent'] is None or i['parent'] > 0 else 1, children=i['sub_regions_ids'],
                    color=i['visualization_data']['color'], broken_stl=broken_stl, flip_normal=(int(k)>1)
                )
        self.get_hierarchy_level(1)

    def _open_h(self, h):
        return [(i['id'], self._open_h(i['children'])) for i in h]

    def get_hierarchy_level(self, level: int):
        ids = [0]
        h = self.hierarchy
        while level:
            ids.extend([i[0] for i in h if not i[1]])
            h = [j for i in h for j in i[1] if i[1]]
            level -= 1
        ids.extend([i[0] for i in h])
        return [self.regions[i] for i in ids]

if __name__ == "__main__":
    at = MPIN_Atlas('/mnt/net/nasdmicro/reference_brain_V2/')
    pass