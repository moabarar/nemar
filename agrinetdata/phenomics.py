import json

import pandas as pd
import requests

'''
use:
p=Phenomics()
df1=p.get_dictioneries()

#long time
dftemp=sum([p.get_dictionary(x)[x].values.tolist() for x in df1.dictionary_name.values.tolist()],[])
temp=[]
for rec in dftemp:
    for typ in set(rec.keys()).difference(set(['record_id','parent'])):
        temp.append({'record_id':rec['record_id'],'field':typ,'value':rec[typ]})
pd.DataFrame(temp)
'''

import base64


def encode_base64(mystr):
    return base64.encodebytes(bytes(mystr, 'utf-8'))


def decode_(mystr):
    return base64.decodebytes(mystr).decode("utf-8")


def auth(username, password):
    url = 'https://www.agri-net.org.il/api/auth/'

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    data = json.dumps({"username": username, "password": password})

    r = requests.post(url, headers=headers, data=data)
    if r.status_code != 200:
        return 'wrong password'
    try:
        res = json.loads(r.text)['access_token']
    except:
        res = 'wrong password'
    return res


class Phenomics(object):

    def verify_auth(self):

        url = 'https://www.agri-net.org.il/api/auth/verify'

        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads((r.text))['oppstatus']

    def __init__(self, is_dev_environemnt=False):
        self.is_dev_environemnt = is_dev_environemnt
        pass

    def login(self, username, password):
        self.auth = auth(username, password)
        self.username = username
        self.password = password
        self.logined = self.verify_auth() == 'Ok'
        self.tried_once_already = False
        print('authentication success: ', self.verify_auth())

    def renew_auth(self):
        try:
            self.auth = auth(self.username, self.password)
            return True
        except:
            print('user login problem!')
            return False

    def check_renew_auth(self):
        try:
            if self.verify_auth() == 'Ok':
                return True
        except:
            return self.renew_auth()

    def get_users(self):

        url = 'https://www.agri-net.org.il/api/auth/listusers'

        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads((r.text))

    def get_dictioneries(self):

        url = 'https://www.agri-net.org.il/api/get_dictionaries/'

        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return pd.DataFrame(json.loads((r.text)))

    def get_experiments(self):

        url = 'https://www.agri-net.org.il/api/get_experiments/'

        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return pd.DataFrame(json.loads((r.text)))

    def get_experiments_data(self):

        url = 'https://www.agri-net.org.il/api/get_experiments_data/'

        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return pd.DataFrame(json.loads((r.text)))

    def get_imaging_tasks(self):

        url = 'https://www.agri-net.org.il/api/get_imaging_tasks/'

        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    #         return pd.DataFrame(json.loads((r.text)))  ---- future change when return format is clear

    def get_dictionary(self, dictionary_name):
        url = 'https://www.agri-net.org.il/api/get_dictionary_data/?dictionary_name=' + dictionary_name

        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return pd.DataFrame(json.loads(r.text))

    def get_experiment_data(self, experiment_id):
        url = 'https://www.agri-net.org.il/api/get_experiment_data/?experiment_id=' + experiment_id

        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_experiment_map(self, experiment_id):
        url = 'https://www.agri-net.org.il/api/get_experiment_map/?experiment_id=' + experiment_id

        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_image_data(self, image_id):
        url = 'https://www.agri-net.org.il/api/get_image_data/?image_id=' + image_id

        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_image_data_by_inner_key(self, image_id, inner_key):
        url = f'https://www.agri-net.org.il/api/get_image_data_by_inner_key/?image_id={image_id}&inner_key={inner_key}'

        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        if self.is_dev_environemnt: url = url + '&dev=true'
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_image_data_by_section(self, image_id, section):
        url = f'https://www.agri-net.org.il/api/get_image_data_by_section/?image_id={image_id}&section={section}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_images_by_camera(self, camera_type, record_id=None):
        if record_id is None:
            url = f'https://www.agri-net.org.il/api/get_images_by_camera/?camera_type={camera_type}'
        else:
            url = f'https://www.agri-net.org.il/api/get_images_by_camera/?camera_type={camera_type}&record_id={record_id}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_images_by_frame(self, fid):
        url = f'https://www.agri-net.org.il/api/get_images_by_frame/?frame_id={fid}&detail_level=detailed'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_images_by_experiment_id(self, experiment_id):
        url = f'https://www.agri-net.org.il/api/get_images_by_experiment_id/?experiment_id={experiment_id}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_images_by_growth_stage(self, growth_stage):
        url = f'https://www.agri-net.org.il/api/get_images_by_growth_stage/?growth_stage={growth_stage}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_images_by_location(self, location_name, record_id):
        url = f'https://www.agri-net.org.il/api/get_images_by_location/?location_name={location_name}&record_id={record_id}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_images_by_organ(self, organ_name, record_id):
        url = f'https://www.agri-net.org.il/api/get_images_by_organ/?organ_name={organ_name}&record_id={record_id}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_images_by_organism(self, inner_organism_name, scientific_name, record_id):
        url = f'https://www.agri-net.org.il/api/get_images_by_organism/?inner_organism_name={inner_organism_name}&scientific_name={scientific_name}&record_id={record_id}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_images_by_plot_name(self, plot_name, experiment_id):
        url = f'https://www.agri-net.org.il/api/get_images_by_plot_name/?plot_name={plot_name}&experiment_id={experiment_id}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_images_by_task_id(self, task_id):
        url = f'https://www.agri-net.org.il/api/get_images_by_task_id/?task_id={task_id}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_files_in_dir_by_id(self, image_id):
        url = f'https://www.agri-net.org.il/api/get_files_in_dir_by_id/?image_id={image_id}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_images_by_treatment(self, task_id):
        url = f'https://www.agri-net.org.il/api/get_images_by_treatment/?treatment_name={treatment_name}&record_id={record_id}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_images_by_variety(self, variety, record_id):
        url = f'https://www.agri-net.org.il/api/get_images_by_variety/?variety={variety}&record_id={record_id}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_images_uploads_data(self):
        url = f'https://www.agri-net.org.il/api/get_images_uploads_data/'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_imaging_protocols(self):
        url = f'https://www.agri-net.org.il/api/get_imaging_protocols/'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_imaging_protocol_by_id(self, protocol_id):
        url = f'https://www.agri-net.org.il/api/get_imaging_protocol_by_id/?protocol_id={protocol_id}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def get_imaging_task_by_id(self, task_id):
        url = f'https://www.agri-net.org.il/api/get_imaging_task_by_id/?task_id={task_id}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

        def get_frames_by_imaging_task_id(self, task_id):
            url = f'https://www.agri-net.org.il/api/get_frames_by_imaging_task_id/?imaging_task_id={task_id}&detail_level=detailed'

        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    # https://www.agri-net.org.il/api/set_image_data/?image_id=43&section=annotator&json_data=bla
    def post_set_image_data(self, image_id, json_data, section='annotator'):
        server = f'https://www.agri-net.org.il/api/set_image_data_by_section/?image_id={image_id}&section={section}&json_data={json_data}'
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   'Authorization': 'JWT ' + self.auth}
        r = requests.post(server, headers=headers)
        return r

    #     def post_set_image_data_by_inner_key(self,image_id,json_data,inner_key):
    #         server = 'https://www.agri-net.org.il/api/set_image_data_by_inner_key/?image_id={image_id}&inner_key={inner_key}&json_data={json_data}'
    #         headers={ "Content-Type" : "application/json", "Accept" : "application/json",'Authorization':'JWT '+self.auth}
    #         r = requests.post(server, headers=headers)
    #         return r

    def get_image_id_from_uri(self, image_uri):
        url = f'https://www.agri-net.org.il/api/get_image_id_from_uri/?image_uri={image_uri}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return r

    def get_frame_metadata(self, frame_id):
        url = f'https://www.agri-net.org.il/api/get_frame_metadata/?frame_id={frame_id}'
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    def post_set_image_data_by_inner_key(self, image_id, json_data, inner_key):
        url = 'https://www.agri-net.org.il/api/set_image_data_by_inner_key/'
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   'Authorization': 'JWT ' + self.auth}
        params = {'image_id': image_id, 'inner_key': inner_key, 'json_data': json_data}
        if self.is_dev_environemnt:
            params['dev'] = 'true'
        else:
            params['dev'] = 'false'
        r = requests.post(url, headers=headers, json=params)
        return r

    # annotation

    def get_annotation_tasks(self, **kwargs):
        '''
        filter annotation tasks by:
        annotation_task_id
        group_name
        experiment_id
        imaging_task_id
        creator
        target
        annotator	
        '''
        url = 'https://www.agri-net.org.il/api/get_annotation_tasks/'
        empty = '?'
        for key, value in kwargs.items():
            empty += f'{key}={value}&'
        empty = empty[:-1]
        url = url + empty
        params = {'Authorization': 'JWT ' + self.auth, 'Accept': 'application/json'}
        r = requests.get(url, headers=params)
        return json.loads(r.text)

    #     def post_set_image_data_by_inner_key(self,image_id, json_data, inner_key):
    #         url = 'https://www.agri-net.org.il/api/set_image_data_by_inner_key/'
    #         headers={ "Content-Type" : "application/json", "Accept" : "application/json",'Authorization':'JWT '+self.auth}
    #         j = {'image_id': image_id, 'inner_key': inner_key, 'json_data': json_data}
    #         if  self.is_dev_environemnt: j['dev'] = 'true'

    #         r = requests.post(url, headers=headers,json=j)
    #         return r

    # https://www.agri-net.org.il/api/get_images_by_camera/?camera_type=Multispectral&record_id=0
    #  https://www.agri-net.org.il/api/get_images_by_experiment_id/?experiment_id=0
    # GET /get_images_by_growth_stage/growth_stage&record_id
    # GET /GET /get_images_by_location/location_name&record_id
    # GET /GET /GET /get_images_by_organ/organ_name&record_id
    #   GET /get_images_by_organism/inner_organism_name&scientific_name&record_id
    # GET /get_images_by_plot_name/plot_name&experiment_id
    #    GET /get_images_by_task_id/task_id
    #   GET /get_images_by_treatment/treatment_name&record_id
    #    GET /get_images_by_variety/variety&record_id
    #    GET /get_images_uploads_data()
    #    GET /get_imaging_protocols
    #    GET /get_imaging_protocol_by_id/protocol_id
    #    GET /get_imaging_task_by_id/task_id
    ''''''

#     def dictionary_proc(self,dictionary):
#         temp=[]
#         for rec in self.get_dictionary(dictionary)[dictionary].values.tolist():
#             for typ in set(rec.keys()).difference(set(['record_id','parent'])):
#                 temp.append({'record_id':rec['record_id'],'field':typ,'value':rec[typ]})
#         #tags_update(temp.astype(str))
#         return temp
#         #self.tags.update(temp)
#     def check_dictionary_update(self):
#         '''future function to check if dictionary is updated
#         should be:
#         if get_update_date()==self.date:
#            return True
#         return False
#         True- is updated
#         '''
#         return False


#     def download_dictionaries(self):

#     # We can use a with statement to ensure threads are cleaned up promptly
#         if not self.check_dictionary_update():    
#             try:
#                 dictionaries
#             except:
#                 dictionaries=self.get_dictioneries()            
#             mylist=[]
#             with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#                 # Start the load operations and mark each future with its URL
#                 future_to_url = {executor.submit(self.dictionary_proc, x): x for x in dictionaries.dictionary_name.unique().tolist()}

#                 for future in concurrent.futures.as_completed(future_to_url):
#                     url = future_to_url[future]
#                     data = future.result()
#                     mylist.append(data)
#             self.tags=pd.DataFrame(sum(mylist,[]))
