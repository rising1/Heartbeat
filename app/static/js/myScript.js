  document.getElementById("prediction").innerHTML = "awaited";

    var dragHandler = function(evt){
        evt.preventDefault();
    };

    var dropHandler = function(evt){
        evt.preventDefault();
        var files = evt.originalEvent.dataTransfer.files;
        var imageout = new Image();

        var reader  = new FileReader();

        // it's onload event and you forgot (parameters)

        reader.onload = function(e)  {
            //var image = document.createElement("img");
            // the result image data
            //image.src = e.target.result;
            var image = document.createElement("img");
	    image.src = e.target.result;
            // the result image data
            console.log(this.width)
            imgRatio = this.height / this.width
            console.log(imgRatio)
            const widthout = 200;
            const heightout  = parseInt(imgRatio * widthout);
            const elem = document.createElement('canvas');
            elem.width = widthout;
            elem.height = heightout;
            const ctx = elem.getContext('2d');
                    // img.width and img.height will contain the original dimensions
            ctx.drawImage(img, 0, 0, widthout, heightout);
            imageout.src = elem.toDataURL("image/png")
            reader.onerror = error => console.log(error);
            $("#birdpic").empty().append(imageout);
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

