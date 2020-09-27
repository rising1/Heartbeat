    document.getElementById("prediction").innerHTML = "awaited";

    var dragHandler = function(evt){
        evt.preventDefault();
    };

    var dropHandler = function(evt){
        evt.preventDefault();
        var files = evt.originalEvent.dataTransfer.files;


        var reader  = new FileReader();

        // it's onload event and you forgot (parameters)

        reader.onload = function(e)  {
            var image = document.createElement("img");
            // the result image data
            image.src = e.target.result;
            document.getElementById("prediction").appendChild(image);
        }

        reader.readAsDataURL(files[0]);


        var formData = new FormData();
        formData.append("file2upload", files[0]);

        var req = {
            url: "/uploader",
            //url: "/test",
            // url: "/sendfile",
            method: "post",
            processData: false,
            contentType: false,
            data: formData
        };
        console.log('Posted');

        // $.ajax(req)
         $.ajax(req)
            .done(function(data){
                $('#prediction').text(data).show();
            } );

         event.preventDefault();
    };

    var dropHandlerSet = {
        dragover: dragHandler,
        drop: dropHandler
    };

    $(".droparea").on(dropHandlerSet);

    // to display the image in the box

    function myFunction() {

        var file = document.getElementById('file').files[0];

        // you have to declare the file loading
        reader.readAsDataURL(file);
    }

