from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

def prediction(lst):
    filename = 'model/predictor.pickle'
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    pred_value = model.predict([lst])
    return pred_value


@app.route('/', methods=['POST','GET'])
def index():
    pred_value = 0

    ram = request.get_json().get("ram")
    weight = request.get_json().get("weight")
    company = request.get_json().get("company")
    typename = request.get_json().get("typename")
    opsys = request.get_json().get("opsys")
    cpu = request.get_json().get("cpu")
    gpu = request.get_json().get("gpu")
    touchscreen = request.get_json().get("touchscreen")
    ips = request.get_json().get("ips")

    feature_list = []

    feature_list.append(int(ram))
    feature_list.append(float(weight))
    feature_list.append(len(touchscreen))
    feature_list.append(len(ips))

    company_list = ['acer','apple','asus','dell','hp','lenovo','msi','other','toshiba']
    typename_list = ['2in1convertible','gaming','netbook','notebook','ultrabook','workstation']
    opsys_list = ['linux','mac','other','windows']
    cpu_list = ['amd','intelcorei3','intelcorei5','intelcorei7','other']
    gpu_list = ['amd','intel','nvidia']



    def traverse_list(lst, value):
        for item in lst:
            if item == value:
                feature_list.append(1)
            else:
                feature_list.append(0)

    traverse_list(company_list, company)
    traverse_list(typename_list, typename)
    traverse_list(opsys_list, opsys)
    traverse_list(cpu_list, cpu)
    traverse_list(gpu_list, gpu)

    pred_value = prediction(feature_list)
    pred_value = np.round(pred_value[0],2)*300
    val = {"prediction": pred_value}
    return jsonify(val)




if __name__ == '__main__':
    app.run(debug=True)
