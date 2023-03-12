function[] = visual_sample(img, bboxs)
    figure()
    imshow(img)
    for i = 1:size(bboxs, 1)
        rectangle('Position',bboxs(i,:));
    end
end