import os
def parse(parser):
    # Parse and set parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written. The directory will be created if it doesn't exist.")
    parser.add_argument("--output_folder", default=None, type=str, required=True,
                        help="The output folder name under the output directory.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: roberta-large, simcse")
    parser.add_argument("--freeze_adapter", default=False, type=bool,
                        help="freeze the parameters of adapter.")
    parser.add_argument("--freeze_bert", default=False, type=bool,
                        help="freeze the parameters of the roberta-large.")
    parser.add_argument("--adapter_transformer_layers", default=2, type=int,
                        help="The transformer layers of adapter.")
    parser.add_argument("--adapter_size", default=768, type=int,
                        help="The hidden size of adapter.")
    parser.add_argument("--adapter_list", default="0,11,22", type=str,
                        help="The layer where add an adapter")
    parser.add_argument("--adapter_skip_layers", default=3, type=int,
                        help="The skip_layers of adapter according to bert layers")
    parser.add_argument('--meta_fac_adaptermodel', default='',type=str, help='the pretrained factual adapter model')
    parser.add_argument('--meta_et_adaptermodel', default='',type=str, help='the pretrained entity typing adapter model')
    parser.add_argument('--meta_lin_adaptermodel', default='', type=str, help='the pretrained linguistic adapter model')
    parser.add_argument('--meta_bertmodel', default='', type=str, help='the pretrained bert model')
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps_cross', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass for cross-encoders.")
    parser.add_argument('--gradient_accumulation_steps_bi', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass for bi-encoders.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--a_rate', type=float, default=0.5,
                        help="Rate of pre-trained LM loss")
    parser.add_argument('--b_rate', type=float, default=0.5,
                        help="Rate of adapter loss")
    parser.add_argument('--metrics', type=str, default='accuracy',
                        help="Metrics to determine the best model")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--normalize', type=bool, default=True,
                        help="Whether to apply normalization before cosine similarity measurement in the bi-encoder setting")
    parser.add_argument("--cycles", type=int, default=1,
                        help="Number of iterations for kowledge distillation. Defaults to 1.")
    parser.add_argument('--fusion_mode', type=str, default='concat',help='the fusion mode for bert feautre (and adapter feature) |add|concat|attentiom')
    parser.add_argument('--loss', type=str, default='bce',
                        help="Loss function. Input one of (bce, infonce, mse)")
    # parser.add_argument('--sim_measure', type=str, default='cosine',
    #                     help="Similarity measure, used only when mode = 'bi'. Input one of (cosine, linear_transform)")
    
    args = parser.parse_args()
    args.adapter_list = args.adapter_list.split(',')
    args.adapter_list = [int(i) for i in args.adapter_list]
    args.my_model_name = args.output_folder
    return args