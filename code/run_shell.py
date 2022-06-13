import os
# os.system("python run_gbert.py --model_name GBert-predict-duikang_epoch=10_p_k_4 "
#           "--num_train_epochs 2 "
#           "--do_train")
# os.system("python run_pretraining.py --model_name GBert-pretraining-duikang_epoch=10_p "
#           "--num_train_epochs 10 --do_train")
# os.system("python run_gbert.py --model_name GBert-predict-duikang_epoch=10_p --use_pretrain "
#           "--pretrain_dir ../saved/GBert-pretraining-duikang_epoch=10_p --num_train_epochs 10 --do_train")

# os.system("python run_pretraining.py --model_name GBert-pretraining-duikang_epoch=10_no-k_1 "
#           "--num_train_epochs 10 --do_train")
# os.system("python run_gbert.py --model_name GBert-predict-duikang_epoch=10_no-k_1 "
#           "--use_pretrain --pretrain_dir ../saved/GBert-pretraining-duikang_epoch=10_no-k_1 "
#           "--num_train_epochs 10 --do_train")
# for i in range(9):
#     print("i="+str(i))
#     os.system("python run_pretraining.py --model_name GBert-pretraining-duikang_epoch=10_no-k_1 "
#               "--use_pretrain --pretrain_dir ../saved/GBert-predict-duikang_epoch=10_no-k_1 "
#               "--num_train_epochs 10 --do_train")
#     os.system("python run_gbert.py --model_name GBert-predict-duikang_epoch=10_no-k_1 "
#               "--use_pretrain --pretrain_dir ../saved/GBert-pretraining-duikang_epoch=10_no-k_1 "
#               "--num_train_epochs 10 --do_train")

for i in range(1):
    # os.system("python run_pretraining.py --model_name GBert-pretraining-wuduikang_epoch=10_no-k-a "
    #             "--num_train_epochs 5 --do_train")
    os.system("python run_pretraining.py --model_name GBert-pretraining-wuduikang_epoch=10_no-k-a "
              "--use_pretrain --pretrain_dir ../saved/GBert-predict-wuduikang_epoch=10_no-k-a "
              "--num_train_epochs 10 --do_train")
    os.system("python run_gbert.py --model_name GBert-predict-wuduikang_epoch=10_no-k-a "
              "--use_pretrain --pretrain_dir ../saved/GBert-pretraining-wuduikang_epoch=10_no-k-a "
              "--num_train_epochs 10 --do_train")
    # os.system("python run_pretraining.py --model_name GBert-pretraining-wuduikang_epoch=10_no-k-a "
    #           "--use_pretrain --pretrain_dir ../saved/GBert-predict-wuduikang_epoch=10_no-k-a "
    #           "--num_train_epochs 10 --do_train --graph")
    # os.system("python run_gbert.py --model_name GBert-predict-wuduikang_epoch=10_no-k-a "
    #           "--num_train_epochs 5 --do_train")
    # os.system("python run_gbert.py --model_name GBert-predict-duikang_epoch=10_p --use_pretrain "
    #           "--pretrain_dir ../saved/GBert-pretraining-duikang_epoch=10_p --num_train_epochs 10 --graph")
# os.system("python run_gbert.py --model_name GBert-predict-duikang3 --use_pretrain "
#             "--pretrain_dir ../saved/GBert-pretraining-duikang3 --num_train_epochs 5 "
#             "--do_train --graph")
# os.system("python run_pretraining.py --model_name GBert-pretraining-duikang4 "
#             "--use_pretrain --pretrain_dir ../saved/GBert-predict-duikang4 "
#             "--num_train_epochs 10 --do_train --graph")
# for i in range(5):
#     print("i=" + str(i))
#     os.system("python run_pretraining.py --model_name GBert-pretraining-duikang4 "
#                 "--use_pretrain --pretrain_dir ../saved/GBert-predict-duikang4 "
#                 "--num_train_epochs 5 --do_train --graph")
#     os.system("python run_gbert.py --model_name GBert-predict-duikang4 --use_pretrain "
#             "--pretrain_dir ../saved/GBert-pretraining-duikang4 --num_train_epochs 5 "
#             "--do_train --graph")
# os.system("python run_gbert.py --model_name GBert-predict-duikang4 --use_pretrain "
#             "--pretrain_dir ../saved/GBert-pretraining-duikang4 --num_train_epochs 5 "
#             "--do_train --graph")
# # os.system("python run_gbert.py --model_name GBert-predict-duikang4 --use_pretrain "
#         "--pretrain_dir ../saved/GBert-pretraining-duikang4 --num_train_epochs 5 "
#         "--do_train --graph")
# os.system("python run_gbert.py --model_name GBert-predict-pretrain-new "
#           "--num_train_epochs 80 "
#           "--do_train --graph")
# os.system("python run_gbert.py --model_name GBert-predict-pretrain-graph "
#           "--num_train_epochs 80 "
#           "--do_train")
# os.system("python run_pretraining.py --model_name GBert-pretraining-graph "
#           "--num_train_epochs 5 --do_train")
# os.system("python run_gbert.py --model_name GBert-predict-graph --use_pretrain "
#           "--pretrain_dir ../saved/GBert-pretraining-graph --num_train_epochs 5 "
#           "--do_train")
# for i in range(15):
#     os.system("python run_pretraining.py --model_name GBert-pretraining-graph "
#               "--use_pretrain --pretrain_dir ../saved/GBert-predict-graph "
#               "--num_train_epochs 5 --do_train")
#     os.system("python run_gbert.py --model_name GBert-predict-graph --use_pretrain "
#               "--pretrain_dir ../saved/GBert-pretraining-graph --num_train_epochs 5 "
#               "--do_train")