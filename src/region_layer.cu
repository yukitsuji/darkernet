#include "cuda_runtime"
#include "curand.h"
#include "cublas_v2"

extern "C" {
#include "region_layer.h"
}

__device__ void entry_index_gpu() {
  // entry_index(layer l, int batch, int location, int entry)
  // int n =   location / (l.w*l.h);
  // int loc = location % (l.w*l.h);
  // return n * l.w * l.h * (l.coords+l.classes+1) + entry * l.w * l.h + loc;
}

__global__ void region_boxes_kernel(layer l, int w, int h,
  int netw, int neth, float thresh, float **probs, box *boxes,
  int only_objectness, int *map, float tree_thresh, int relative,
  int wh) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // int n =

  // unsigned int ix =
  // unsigned int iy =
  // unsigned int iz =
}

void forward_getregion_boxes_gpu(layer l, int w, int h,
  int netw, int neth, float thresh, float **probs, box *boxes,
  int only_objectness, int *map, float tree_thresh, int relative) {

  dim3 block(512);
  int size = (l.batch * l.w * l.h * l.n / block.x - 1) / block.x
  dim3 grid(size);
  int wh = w * h;

  region_boxes_kernel<<< grid, block >>>(l, w, h,
    netw, neth, thresh, probs, box boxes,
    only_objectness, map, tree_thresh, relative,
    wh);
}

//
// void my_forward_region_layer_gpu(const layer l, network net)
// {
//     int i,j,b,t,n;
//     memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
//
// #ifndef GPU
//     for (b = 0; b < l.batch; ++b){
//         for(n = 0; n < l.n; ++n){
//             int index = entry_index(l, b, n*l.w*l.h, 0);
//             activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
//             index = entry_index(l, b, n*l.w*l.h, 4);
//             if(!l.background) activate_array(l.output + index,   l.w*l.h, LOGISTIC);
//         }
//     }
//     if (l.softmax){
//         int index = entry_index(l, 0, 0, l.coords + !l.background);
//         softmax_cpu(net.input + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
//     }
// #endif
//
//     memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
//     if(!net.train) return;
//     float avg_iou = 0;
//     float recall = 0;
//     float avg_cat = 0;
//     float avg_obj = 0;
//     float avg_anyobj = 0;
//     int count = 0;
//     int class_count = 0;
//     *(l.cost) = 0;
//     for (b = 0; b < l.batch; ++b) {
//         for (j = 0; j < l.h; ++j) {
//             for (i = 0; i < l.w; ++i) {
//                 for (n = 0; n < l.n; ++n) {
//                     int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
//                     box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
//                     float best_iou = 0;
//                     for(t = 0; t < 30; ++t){
//                         box truth = float_to_box(net.truth + t*5 + b*l.truths, 1);
//                         if(!truth.x) break;
//                         float iou = box_iou(pred, truth);
//                         if (iou > best_iou) {
//                             best_iou = iou;
//                         }
//                     }
//                     int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
//                     avg_anyobj += l.output[obj_index];
//                     l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
//                     if(l.background) l.delta[obj_index] = l.noobject_scale * (1 - l.output[obj_index]);
//                     if (best_iou > l.thresh) {
//                         l.delta[obj_index] = 0;
//                     }
//
//                     if(*(net.seen) < 12800){
//                         box truth = {0};
//                         truth.x = (i + .5)/l.w;
//                         truth.y = (j + .5)/l.h;
//                         truth.w = l.biases[2*n]/l.w;
//                         truth.h = l.biases[2*n+1]/l.h;
//                         delta_region_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, .01, l.w*l.h);
//                     }
//                 }
//             }
//         }
//         for(t = 0; t < 30; ++t){
//             box truth = float_to_box(net.truth + t*5 + b*l.truths, 1);
//
//             if(!truth.x) break;
//             float best_iou = 0;
//             int best_n = 0;
//             i = (truth.x * l.w);
//             j = (truth.y * l.h);
//             //printf("%d %f %d %f\n", i, truth.x*l.w, j, truth.y*l.h);
//             box truth_shift = truth;
//             truth_shift.x = 0;
//             truth_shift.y = 0;
//             //printf("index %d %d\n",i, j);
//             for(n = 0; n < l.n; ++n){
//                 int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
//                 box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
//                 if(l.bias_match){
//                     pred.w = l.biases[2*n]/l.w;
//                     pred.h = l.biases[2*n+1]/l.h;
//                 }
//                 //printf("pred: (%f, %f) %f x %f\n", pred.x, pred.y, pred.w, pred.h);
//                 pred.x = 0;
//                 pred.y = 0;
//                 float iou = box_iou(pred, truth_shift);
//                 if (iou > best_iou){
//                     best_iou = iou;
//                     best_n = n;
//                 }
//             }
//             //printf("%d %f (%f, %f) %f x %f\n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h);
//
//             int box_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 0);
//             float iou = delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
//             if(iou > .5) recall += 1;
//             avg_iou += iou;
//
//             //l.delta[best_index + 4] = iou - l.output[best_index + 4];
//             int obj_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords);
//             avg_obj += l.output[obj_index];
//             l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
//             if (l.rescore) {
//                 l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
//             }
//             if(l.background){
//                 l.delta[obj_index] = l.object_scale * (0 - l.output[obj_index]);
//             }
//
//             int class = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
//             if (l.map) class = l.map[class];
//             int class_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords + 1);
//             delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat);
//             ++count;
//             ++class_count;
//         }
//     }
//     //printf("\n");
//     *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
//     printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
// }
