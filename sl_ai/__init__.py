import ffmpeg


def ffmpeg_resize(vid_path, output_path, width, height):
    """
    use ffmpeg to resize the input person_video to the width given, keeping aspect ratio
    TODO: Is this still required? Maybe delete it.
    """
    input_vid = ffmpeg.input(vid_path, an=None)
    vid = (
        input_vid.filter("scale", width, height)
        .output(output_path, loglevel="quiet")
        .overwrite_output()
        .run()
    )
    return output_path
