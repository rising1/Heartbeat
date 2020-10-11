

    var dragHandler = function(evt){
        evt.preventDefault();
    };

    var dropHandler = function(evt){
        var image = document.createElement("img");
        evt.preventDefault();
        var files = evt.originalEvent.dataTransfer.files;
        var reader  = new FileReader();

        reader.onload = function(e)  {
            image.src = e.target.result;
            image.addEventListener("load",function(){
                console.log('image has loaded')
                console.log(image.width + " "  + image.height)
                imgRatio = image.height / image.width
                const widthout = 200;
                const heightout  = parseInt(imgRatio * widthout);
                const elem = document.createElement('canvas');
                elem.width = widthout;
                elem.height = heightout;
                const ctx = elem.getContext('2d');
                ctx.drawImage(image, 0, 0, widthout, heightout);
                var dataurl = elem.toDataURL("image/jpeg");
                var newimage = document.createElement("img");
                newimage.src = dataurl
                $("#birdpic").empty().append(newimage);
                //$("#birdpic").empty().append(image);
            })
        }

        reader.onerror = error => console.log(error);

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
                results = data.split(",")
                $('#prediction1').html("<h3>" + results[0] + "</h3> \n -score " +
                    results[1]).show();
                $('#prediction2').text(results[2] + " -score " +
                    results[3]).show();
                $('#prediction3').text(results[4] + " -score " +
                    results[5]).show();
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

