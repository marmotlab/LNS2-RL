import os
import os.path as osp

import numpy as np
import ray
import setproctitle
import torch
import wandb

from alg_parameters import *
from model import Model
from runner import RLRunner
from util import set_global_seeds, write_to_wandb, perf_dict_driver,write_to_wandb_im

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ray.init(num_gpus=SetupParameters.NUM_GPU)
print("Welcome to Dynamic MAPF!\n")


def main():
    """main code"""
    # preparing for training
    if RecordingParameters.RETRAIN:
        restore_path = ''
        net_path_checkpoint = restore_path + "/net_checkpoint.pkl"
        net_dict = torch.load(net_path_checkpoint)

    if RecordingParameters.WANDB:
        if RecordingParameters.RETRAIN:
            wandb_id = ''
        else:
            wandb_id = wandb.util.generate_id()
        wandb.init(project=RecordingParameters.EXPERIMENT_PROJECT,
                   name=RecordingParameters.EXPERIMENT_NAME,
                   entity=RecordingParameters.ENTITY,
                   notes=RecordingParameters.EXPERIMENT_NOTE,
                   config=all_args,
                   id=wandb_id,
                   resume='allow')
        print('id is:{}'.format(wandb_id))
        print('Launching wandb...\n')

    setproctitle.setproctitle(
        RecordingParameters.EXPERIMENT_PROJECT + RecordingParameters.EXPERIMENT_NAME + "@" + RecordingParameters.ENTITY)
    set_global_seeds(SetupParameters.SEED)

    # create classes
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
    global_model = Model(0, global_device, True)

    if RecordingParameters.RETRAIN:
        global_model.network.load_state_dict(net_dict['model'])
        global_model.net_optimizer.load_state_dict(net_dict['optimizer'])

    envs = [RLRunner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]

    if RecordingParameters.RETRAIN:
        curr_steps = net_dict["step"]
        curr_episodes = net_dict["episode"]
        im_steps=net_dict["im_step"]
    else:
        curr_steps = curr_episodes =im_steps = 0

    update_done = True
    job_list = []
    last_model_t = -RecordingParameters.SAVE_INTERVAL - 1
    last_print_t = -RecordingParameters.PRINT_INTERVAL - 1
    cl_task_num=0
    prev_cl_task_num=0
    demon = False
    # start training
    try:
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            if update_done:
                # start a data collection
                switch_task = False
                if global_device != local_device:
                    net_weights = global_model.network.to(local_device).state_dict()
                    global_model.network.to(global_device)
                else:
                    net_weights = global_model.network.state_dict()
                net_weights_id = ray.put(net_weights)

                if curr_steps>=EnvParameters.SWITCH_TIMESTEP[0] and curr_steps<EnvParameters.SWITCH_TIMESTEP[1]:
                    cl_task_num = 1
                if curr_steps>=EnvParameters.SWITCH_TIMESTEP[1]:
                    cl_task_num = 2

                if cl_task_num-prev_cl_task_num>0:
                    switch_task=True
                    im_steps=0
                    print("switch to task {}".format(cl_task_num))
                prev_cl_task_num=cl_task_num

                cl_task_num_id= ray.put(cl_task_num)
                switch_task_id= ray.put(switch_task)

                if im_steps < TrainingParameters.DEMONSTRATION_THRES[cl_task_num]:
                    demon = True
                    for i, env in enumerate(envs):
                        job_list.append(env.im_run.remote(net_weights_id,cl_task_num_id,switch_task_id))
                else:
                    demon = False
                    for i, env in enumerate(envs):
                        job_list.append(env.rl_run.remote(net_weights_id,cl_task_num_id,switch_task_id))

            # get data from multiple processes
            done_id, job_list = ray.wait(job_list, num_returns=TrainingParameters.N_ENVS)
            update_done = True if job_list == [] else False
            done_len = len(done_id)
            job_results = ray.get(done_id)

            if demon:
                # get imitation learning data
                im_data_buffer = {"obs": [], "vector": [], "action": [], "hidden_state": []}
                tem_step=0
                for results in range(done_len):
                    for i, key in enumerate(im_data_buffer.keys()):
                        im_data_buffer[key].append(job_results[results][i])
                    curr_episodes += job_results[results][-2]
                    tem_step+= job_results[results][-1]

                curr_steps += tem_step
                im_steps+= tem_step
                for key in im_data_buffer.keys():
                    im_data_buffer[key] = np.concatenate(im_data_buffer[key], axis=0)

                # training of imitation learning
                imitation_loss = []
                inds = np.arange(tem_step)
                for _ in range(TrainingParameters.N_EPOCHS):
                    np.random.shuffle(inds)
                    for start in range(0, tem_step, TrainingParameters.MINIBATCH_SIZE):
                        end = start + TrainingParameters.MINIBATCH_SIZE
                        mb_inds = inds[start:end]
                        slices = (arr[mb_inds] for arr in
                                  (im_data_buffer["obs"], im_data_buffer["vector"],
                                   im_data_buffer["action"], im_data_buffer["hidden_state"]))
                        imitation_loss.append(global_model.imitation_train(*slices))

                if RecordingParameters.WANDB:
                    write_to_wandb_im(curr_steps, imitation_loss)
            else:
                curr_steps += done_len * TrainingParameters.N_STEPS
                data_buffer = {"obs": [], "vector": [], "returns": [], "values": [], "action": [], "ps": [],
                               "hidden_state": [], "train_valid": []}
                perf_dict = perf_dict_driver()
                for results in range(done_len):
                    for i, key in enumerate(data_buffer.keys()):
                        data_buffer[key].append(job_results[results][i])
                    curr_episodes +=job_results[results][-2]
                    for key in perf_dict.keys():
                        perf_dict[key].append(np.nanmean(job_results[results][-1][key]))

                for key in data_buffer.keys():
                    data_buffer[key] = np.concatenate(data_buffer[key], axis=0)

                for key in perf_dict.keys():
                    perf_dict[key] = np.nanmean(perf_dict[key])

                # training of reinforcement learning
                mb_loss = []
                inds = np.arange(done_len * TrainingParameters.N_STEPS)
                for _ in range(TrainingParameters.N_EPOCHS):
                    np.random.shuffle(inds)
                    for start in range(0, done_len * TrainingParameters.N_STEPS, TrainingParameters.MINIBATCH_SIZE):
                        end = start + TrainingParameters.MINIBATCH_SIZE
                        mb_inds = inds[start:end]
                        slices = (arr[mb_inds] for arr in
                                  (data_buffer["obs"], data_buffer["vector"],data_buffer["returns"],data_buffer["values"],
                                       data_buffer["action"], data_buffer["ps"],data_buffer["hidden_state"],
                                       data_buffer["train_valid"]))
                        mb_loss.append(global_model.train(*slices))

                # record training result
                if RecordingParameters.WANDB:
                    write_to_wandb(curr_steps, perf_dict, mb_loss)

                if (curr_steps - last_print_t) / RecordingParameters.PRINT_INTERVAL >= 1.0:
                    last_print_t = curr_steps
                    print('episodes: {}, steps: {}, win rate:{} \n'.format(
                        curr_episodes, curr_steps,perf_dict["team_better"]))

            # save model
            if (curr_steps - last_model_t) / RecordingParameters.SAVE_INTERVAL >= 1.0:
                last_model_t = curr_steps
                print('Saving Model !\n')
                model_path = osp.join(RecordingParameters.MODEL_PATH, '%.5i' % curr_steps)
                os.makedirs(model_path)
                path_checkpoint = model_path + "/net_checkpoint.pkl"
                net_checkpoint = {"model": global_model.network.state_dict(),
                                  "optimizer": global_model.net_optimizer.state_dict(),
                                  "step": curr_steps,
                                  "episode": curr_episodes,
                                  "im_step": im_steps}
                torch.save(net_checkpoint, path_checkpoint)

    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")
    finally:
        # save final model
        print('Saving Final Model !\n')
        model_path = RecordingParameters.MODEL_PATH + '/final'
        os.makedirs(model_path)
        path_checkpoint = model_path + "/net_checkpoint.pkl"
        net_checkpoint = {"model": global_model.network.state_dict(),
                          "optimizer": global_model.net_optimizer.state_dict(),
                          "step": curr_steps,
                          "episode": curr_episodes,
                          "im_step": im_steps}
        torch.save(net_checkpoint, path_checkpoint)
        # killing
        for e in envs:
            ray.kill(e)
        if RecordingParameters.WANDB:
            wandb.finish()


if __name__ == "__main__":
    main()
