const form = document.getElementById("upload-form");
const imageInput = document.getElementById("image");
const previewImg = document.getElementById("preview-img");
const captionText = document.getElementById("caption-text");

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    previewImg.src = URL.createObjectURL(file);
    previewImg.hidden = false;
});

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    captionText.innerText = "Generating caption... ⏳";

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();
    captionText.innerText = data.caption;
});
