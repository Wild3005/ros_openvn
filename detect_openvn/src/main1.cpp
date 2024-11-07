#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <openvino/openvino.hpp>

#include "rclcpp/rclcpp.hpp"
#include "detect_messages/msg/detect_object.hpp"

using namespace std;

const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

cv::VideoCapture cap(2);

class Detect : public rclcpp::Node
{
public:
struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};


struct Resize
{
    cv::Mat resized_image;
    int dw;
    int dh;
};


cv::Mat img;
// publisher_ = this->create_publisher<detect_messages::msg::DetectObject>("detect", 10);
// timer_ = this->create_wall_timer(500ms, std::bind(&MinimalPublisher::timer_callback, this));

ov::CompiledModel compiled_model;
ov::InferRequest infer_request;

detect_messages::msg::DetectObject info_obj;

// Publisher
rclcpp::Publisher<detect_messages::msg::DetectObject>::SharedPtr obj_pub;
// Timer Base
rclcpp::TimerBase::SharedPtr timer_publisher;


Detect(): Node("camera"){
    this->obj_pub = this->create_publisher<detect_messages::msg::DetectObject>("detect_topic", 10);
    this->timer_publisher = this->create_wall_timer(33ms, std::bind(&Detect::detect_callback, this));
    // Step 1. Initialize OpenVINO Runtime core
    ov::Core core;
    // Step 2. Read a model
    std::shared_ptr<ov::Model> model = core.read_model("/home/ichbinwil/openvn_test/yolov5s_openvino_model/yolov5s.xml");
    // Step 4. Inizialize Preprocessing for the model
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    // Specify input image format
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    // Specify preprocess pipeline to input image without resizing
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});
    //  Specify model's input layout
    ppp.input().model().set_layout("NCHW");
    model = ppp.build();
    
    compiled_model = core.compile_model(model, "CPU");
    // Step 6. Create an infer request for model inference 
    infer_request = compiled_model.create_infer_request();

    if(!cap.isOpened()){
        RCLCPP_ERROR(this->get_logger(), "Error opening video stream or file");
        rclcpp::shutdown();
    }

}

private:

Resize resize_and_pad(cv::Mat& img, cv::Size new_shape) {
    float width = img.cols;
    float height = img.rows;
    float r = float(new_shape.width / max(width, height));
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    Resize resize;
    cv::resize(img, resize.resized_image, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_AREA);

    resize.dw = new_shape.width - new_unpadW;
    resize.dh = new_shape.height - new_unpadH; 
    cv::Scalar color = cv::Scalar(100, 100, 100);
    cv::copyMakeBorder(resize.resized_image, resize.resized_image, 0, resize.dh, 0, resize.dw, cv::BORDER_CONSTANT, color);

    return resize;
}

void detect_callback(){
        // if(img.empty()) break;
        // Resize res = resize_and_pad(img, cv::Size(640, 640));
        Resize res;
        cap.read(this -> img);

        res = resize_and_pad(img, cv::Size(640, 640));

        cv::waitKey(1);

        // Step 5. Create tensor from image
        float *input_data = (float *) res.resized_image.data;
        ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        //Step 7. Retrieve inference results 
        const ov::Tensor &output_tensor = infer_request.get_output_tensor();
        ov::Shape output_shape = output_tensor.get_shape();
        float *detections = output_tensor.data<float>();

        
        // Step 8. Postprocessing including NMS  
        std::vector<cv::Rect> boxes;
        vector<int> class_ids;
        vector<float> confidences;

        for (size_t i = 0; i < output_shape[1]; i++){
            float *detection = &detections[i * output_shape[2]];

            float confidence = detection[4];
            if (confidence >= CONFIDENCE_THRESHOLD){
                float *classes_scores = &detection[5];
                cv::Mat scores(1, output_shape[2] - 5, CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;
                cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > SCORE_THRESHOLD){

                    confidences.push_back(confidence);

                    class_ids.push_back(class_id.x);

                    float x = detection[0];
                    float y = detection[1];
                    float w = detection[2];
                    float h = detection[3];

                    float xmin = x - (w / 2);
                    float ymin = y - (h / 2);

                    boxes.push_back(cv::Rect(xmin, ymin, w, h));
                }
            }
        }
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
        std::vector<Detection> output;
        for (size_t i = 0; i < nms_result.size(); i++)
        {
            Detection result;
            int idx = nms_result[i];
            result.class_id = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            output.push_back(result);
        }


        // Step 9. Print results and save Figure with detections
        for (size_t i = 0; i < output.size(); i++)
        {
            auto detection = output[i];
            auto box = detection.box;
            auto classId = detection.class_id;
            auto confidence = detection.confidence;
            float rx = (float)img.cols / (float)(res.resized_image.cols - res.dw);
            float ry = (float)img.rows / (float)(res.resized_image.rows - res.dh);
            box.x = rx * box.x;
            box.y = ry * box.y;
            box.width = rx * box.width;
            box.height = ry * box.height;
            cout << "Bbox" << i + 1 << ": Class: " << classId << " "
                << "Confidence: " << confidence << " Scaled coords: [ "
                << "cx: " << (float)(box.x + (box.width / 2)) / img.cols << ", "
                << "cy: " << (float)(box.y + (box.height / 2)) / img.rows << ", "
                << "w: " << (float)box.width / img.cols << ", "
                << "h: " << (float)box.height / img.rows << " ]" << endl;
            float xmax = box.x + box.width;
            float ymax = box.y + box.height;
            info_obj.center_x = (float)(box.x + (box.width / 2)) / img.cols;
            info_obj.center_y = (float)(box.y + (box.height / 2)) / img.rows;
            cv::rectangle(img, cv::Point(box.x, box.y), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 3);
            cv::rectangle(img, cv::Point(box.x, box.y - 20), cv::Point(xmax, box.y), cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(img, std::to_string(classId), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        obj_pub->publish(this->info_obj);

        // cv::imwrite("detection_cpp.png", img);
        // Tampilkan gambar dengan bounding box di jendela OpenCV
        cv::imshow("Detection Result", img);

}

};



int main(int argc, char **argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Detect>());
  rclcpp::shutdown();
  return 0;
}