# -*- coding:utf-8 -*-

import os
import sys
import argparse
import datetime


import data_road_preprocessing
import model.lstm_order
import model.lstm_click
import model.mlp_order
import model.mlp_click
import model.bagging_order
import model.bagging_click
import model.tpot_order
import model.tpot_click

#########################################################
# argument setting

parser = argparse.ArgumentParser()
parser.add_argument("--mall", help="mall id", nargs='+')
parser.add_argument("--sd", help="start day, 0000-00-00")
parser.add_argument("--ed", help="end day, 0000-00-00")
args = parser.parse_args()

#total_model = ['lstm_order', 'mlp_order', 'bagging_order', 'tpot_order', 'lstm_click', 'mlp_click', 'bagging_click', 'tpot_click']
total_model = ['lstm_order', 'mlp_order', 'bagging_order', 'lstm_click', 'mlp_click', 'bagging_click']

try:
    mall_id = ['chuukr', 'dabainsang', 'hkm0977', 'khyelyun', 'mall66', 'myharoo21', 'planbco', 'sseoqkr7', 'tkddk3704', 'zemmaworld']
    for mall in args.mall:
        if not mall in mall_id:
            raise
except:
    print("confirm mall id")
    sys.exit(1)
    

try:
    start = datetime.datetime.strptime(args.sd, "%Y-%m-%d")
    end = datetime.datetime.strptime(args.ed, "%Y-%m-%d")
    if start > end:
        raise
except:
    print("confirm date")
    sys.exit(1)

    
#########################################################
# 저장할 모델 디렉토리 만들기    
def make_dir(select_mall, start_day, select_model=''):
    
    
    f_path = select_mall + '/' + start_day  + '/' + select_model
    
    try:
        if not(os.path.isdir(f_path)):
            os.makedirs(os.path.join(f_path))
        #else:
        #    print('exist directory')
        #    sys.exit(1)    
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise 
    
    
    return f_path



def info_write(input_config, output_config, infoToFile):
    
    infoToFile.write('----- Hyper parameter -----\n')
    for input_pram in input_config.__dict__:
        #print(input_pram)
        #print(input_pram.__dict__[input_pram])
        infoToFile.write(str(input_pram) + ' : ' + str(input_config.__dict__[input_pram]) + '\n')
    
    infoToFile.write('\n----- Model result -----\n')
    
    sorted_list = sorted(output_config.__dict__)
    
    for output in sorted(output_config.__dict__):
        #print(output)
        #print(output_config.__dict__[output])
        infoToFile.write(str(output) + ' : ' + str(output_config.__dict__[output]) + '\n')
    
    """
    for output in output_config.__dict__:
        #print(output)
        #print(output_config.__dict__[output])
        infoToFile.write(str(output) + ' : ' + str(output_config.__dict__[output]) + '\n')
    """
    infoToFile.write('*'*60 + '\n')


def main():
    print("auto train start")
    
    st_day = datetime.datetime.today().strftime("%Y-%m-%d") 
    
    
    
    
    
    
    for mall in args.mall:
        print('*'*50)
        print(mall + " data preprocessing")
        
        folder_path = make_dir(mall, st_day)
        analysis_path = make_dir(mall, st_day, 'analysis')
        
        road_data = data_road_preprocessing.preprocessing(mall_id=mall, start_day=args.sd, end_day=args.ed, data_analysis_path=analysis_path)
        print(mall + " data preprocessing end")
        
        folder_path = make_dir(mall, st_day)
        analysis_path = make_dir(mall, st_day, 'analysis')
        
        total_info_save = folder_path + 'toal_model_info.txt'
        f_info = open(total_info_save, 'w')
        f_info.write("*"*60 + "\n")
        f_info.write("mall       : %s\n" % mall)
        f_info.write("date range : %s ~ %s\n" % (args.sd, args.ed))
        f_info.write("*"*60 + "\n")
        
        for select_model in total_model:
            
            if select_model == 'lstm_order':
                
                f_info.write("Model name : LSTM order\n\n")
                
                # 모델 폴더 생성
                folder_path = make_dir(mall, st_day, select_model) 
                
                # 모델 config 생성(개별 tunning 및 total info file에 저장하기 위해
                lstm_order_config = model.lstm_order.ModelConfig()
                
                # 모델 output을 config로 출력
                m_output = model.lstm_order.LstmOrder(road_data, folder_path, lstm_order_config)

                
                # 최종 정보 파일에 쓰기(total info에, 전체적으로 모델을 비교하기 위해)
                info_write(lstm_order_config, m_output,f_info)
            
            elif select_model == 'lstm_click':
                
                f_info.write("Model name : LSTM click\n\n")
                
                # 모델 폴더 생성
                folder_path = make_dir(mall, st_day, select_model) 
                
                # 모델 config 생성(개별 tunning 및 total info file에 저장하기 위해
                lstm_click_config = model.lstm_click.ModelConfig()
                
                # 모델 output을 config로 출력
                m_output = model.lstm_click.LstmClick(road_data, folder_path, lstm_click_config)
                
                
                # 최종 정보 파일에 쓰기(total info에, 전체적으로 모델을 비교하기 위해)
                info_write(lstm_click_config, m_output,f_info)
            
            elif select_model == 'mlp_order':
                f_info.write("Model name : MLP order\n\n")
                
                # 모델 폴더 생성
                folder_path = make_dir(mall, st_day, select_model) 
                
                # 모델 config 생성(개별 tunning 및 total info file에 저장하기 위해
                mlp_order_config = model.mlp_order.ModelConfig()
                
                # 모델 output을 config로 출력
                m_output = model.mlp_order.MlpOrder(road_data, folder_path, mlp_order_config)
                
                # 최종 정보 파일에 쓰기(total info에, 전체적으로 모델을 비교하기 위해)
                info_write(mlp_order_config, m_output,f_info)
                
            
            elif select_model == 'mlp_click':
                f_info.write("Model name : MLP click\n\n")
                
                # 모델 폴더 생성
                folder_path = make_dir(mall, st_day, select_model) 
                
                # 모델 config 생성(개별 tunning 및 total info file에 저장하기 위해
                mlp_click_config = model.mlp_click.ModelConfig()
                
                # 모델 output을 config로 출력
                m_output = model.mlp_click.MlpClick(road_data, folder_path, mlp_click_config)
                
                # 최종 정보 파일에 쓰기(total info에, 전체적으로 모델을 비교하기 위해)
                info_write(mlp_click_config, m_output,f_info)
                
                
            elif select_model == 'bagging_order':
                f_info.write("Model name : Bagging order\n\n")
                
                # 모델 폴더 생성
                folder_path = make_dir(mall, st_day, select_model)
                
                # 모델 config 생성(개별 tunning 및 total info file에 저장하기 위해
                bagging_order_config = model.bagging_order.ModelConfig()
                
                # 모델 output을 config로 출력
                m_output = model.bagging_order.BaggingOrder(road_data, folder_path, bagging_order_config)
                
                # 최종 정보 파일에 쓰기(total info에, 전체적으로 모델을 비교하기 위해)
                info_write(bagging_order_config, m_output, f_info)
                
                
            elif select_model == 'bagging_click':
                f_info.write("Model name : Bagging click\n\n")
                
                # 모델 폴더 생성
                folder_path = make_dir(mall, st_day, select_model)
                
                # 모델 config 생성(개별 tunning 및 total info file에 저장하기 위해
                bagging_click_config = model.bagging_click.ModelConfig()
                
                # 모델 output을 config로 출력
                m_output = model.bagging_click.BaggingClick(road_data, folder_path, bagging_click_config)
                
                # 최종 정보 파일에 쓰기(total info에, 전체적으로 모델을 비교하기 위해)
                info_write(bagging_click_config, m_output, f_info)
                
             
            elif select_model == 'tpot_order':
                f_info.write("Model name : TPOT order\n\n")
                
                # 모델 폴더 생성
                folder_path = make_dir(mall, st_day, select_model)
                           
                # 모델 config 생성(개별 tunning 및 total info file에 저장하기 위해
                tpot_order_config = model.tpot_order.ModelConfig()
                
                # 모델 output을 config로 출력
                m_output = model.tpot_order.TpotOrder(road_data, folder_path, tpot_order_config)
                
                # 최종 정보 파일에 쓰기(total info에, 전체적으로 모델을 비교하기 위해)
                info_write(tpot_order_config, m_output, f_info)
                
                
            elif select_model == 'tpot_click':
                f_info.write("Model name : TPOT click\n\n")
                
                # 모델 폴더 생성
                folder_path = make_dir(mall, st_day, select_model)
                           
                # 모델 config 생성(개별 tunning 및 total info file에 저장하기 위해
                tpot_click_config = model.tpot_click.ModelConfig()
                
                # 모델 output을 config로 출력
                m_output = model.tpot_click.TpotClick(road_data, folder_path, tpot_click_config)
                
                # 최종 정보 파일에 쓰기(total info에, 전체적으로 모델을 비교하기 위해)
                info_write(tpot_click_config, m_output, f_info)
        
        f_info.close()
        
        

if __name__ == "__main__":
    main()