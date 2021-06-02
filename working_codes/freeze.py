from keras import backend as K
from keras.models import load_model 
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_NAME="weights-improvement-47-13-0.98"
def export_model(saver,model,input_node_names,output_node_name):
    tf.train.write_graph(K.get_session().graph_def,'out',MODEL_NAME+'graph.pbtxt');
    saver.save(K.get_session(),'out/'+MODEL_NAME+'.chkp')
    freeze_graph.freeze_graph('out/'+MODEL_NAME+'graph.pbtxt',None,False,'out/'+MODEL_NAME+'.chkp',
    output_node_name,"save/restore_all","save/Const:0",'out/frozen_'+MODEL_NAME+'.pb',True,"")
    input_graph_def=tf.GraphDef()
    with tf.gfile.Open('out/frozen_'+MODEL_NAME+'.pb',"rb") as f:
        input_graph_def.ParseFromString(f.read())
    output_graph_def=optimize_for_inference_lib.optimize_for_inference(input_graph_def,input_node_names,
    [output_node_name],tf.float32.as_datatype_enum)
    with tf.gfile.FastGFile('out/frozen'+MODEL_NAME+'.pb','wb') as f:
         f.write(output_graph_def.SerializeToString())
    print("graph saved")
    return
model=load_model('weights-improvement-47-13-0.98.hdf5')
print("output:",model.output.op.name,";",model.input.op.name)

export_model(tf.train.Saver(),model,['input_input'],'op/Softmax')