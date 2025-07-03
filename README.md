# NEURAL-STYLE-TRANSFER

Company : CODTECH IT SOLUTIONS

Name : Pravin S

Intern Id : CT04DG2370

Domain : ARTIFICIAL INTELLIGENCE

Duration : 4 Weeks

Mentor : Neela Santosh

Description:

This Python script performs neural style transfer using PyTorch and a pre-trained VGG19 model. The objective is to generate a new image that combines the content of one image (e.g., a photo) with the artistic style of another (e.g., a painting). The script begins by loading and preprocessing the content and style images, ensuring they are the same size and converting them into tensors. It then defines two custom loss functions: ContentLoss, which measures how much the generated image preserves the structure of the content image, and StyleLoss, which uses Gram matrices to evaluate how well the texture and colors of the style image are replicated. A copy of the VGG19 network is constructed with these loss layers inserted at selected points. The input image, initialized as a clone of the content image, is optimized using the L-BFGS algorithm to minimize a weighted combination of style and content losses over several iterations. Finally, the resulting stylized image is saved and displayed, showcasing a visually appealing blend of the original photo's structure and the painting's artistic feel.

Output:

![Image](https://github.com/user-attachments/assets/2e78952c-8627-4a26-999e-b2708773a31c)
