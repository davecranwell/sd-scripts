python finetune/tag_images_by_wd14_tagger.py --onnx --repo_id "SmilingWolf/wd-vit-tagger-v3" --batch_size 4 --remove_underscore --character_tags_first --character_tag_expand --always_first_tags "trigger_word,class" --undesired_tags "" $1

