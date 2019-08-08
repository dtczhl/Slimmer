% fps first

clear, clc

iou_arr = 0.6:0.02:0.7;
iou_arr = 100 * iou_arr;

fps_arr = linspace(20, 1, length(iou_arr));


text_arr = {'\bullet 1 grid50', '1 grid50', '1 grid50', '1 grid50', '1 grid50', '1 grid50'};

figure(1), clf, hold on
xlabel('FPS')
ylabel('IOU (%)')
xlim([0, max(fps_arr)+5])
ylim([iou_arr(1)-10, iou_arr(end)+10])
for i = 1:length(text_arr)
    txt = text_arr{i};
    h_txt = text(fps_arr(i), iou_arr(i), txt);
    set(h_txt, 'Rotation', 270)
end
box on
grid on
hold off