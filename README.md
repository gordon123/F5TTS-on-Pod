# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching. Support For Thai language.

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/SWivid/F5-TTS)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.06885)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://swivid.github.io/F5-TTS/)
[![hfspace](https://img.shields.io/badge/🤗-Space%20demo-yellow)](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
[![msspace](https://img.shields.io/badge/🤖-Space%20demo-blue)](https://modelscope.cn/studios/modelscope/E2-F5-TTS)
[![lab](https://img.shields.io/badge/X--LANCE-Lab-grey?labelColor=lightgrey)](https://x-lance.sjtu.edu.cn/)
[![lab](https://img.shields.io/badge/Peng%20Cheng-Lab-grey?labelColor=lightgrey)](https://www.pcl.ac.cn)
<!-- <img src="https://github.com/user-attachments/assets/12d7749c-071a-427c-81bf-b87b91def670" alt="Watermark" style="width: 40px; height: auto"> -->

เครื่องมือเปลี่ยนข้อความเป็นคำพูดภาษาไทย Zero-shot TTS ด้วยโมเดล F5-TTS <br>
โมเดล Finetune : [VIZINTZOR/F5-TTS-THAI](https://huggingface.co/VIZINTZOR/F5-TTS-THAI) <br>
 - ชุดข้อมุลในการเทรน : [Porameht/processed-voice-th-169k](https://huggingface.co/datasets/Porameht/processed-voice-th-169k)
 - จำนวน 150,000 เสียง ประมาณ 165 ชั่วโมง
 - โมเดล last steps : 500,000
 - การอ่านข้อความยาวๆ หรือบางคำ ยังไม่ถูกต้อง
# การติดตั้ง บน RUNPOD.io
```sh
git clone https://github.com/gordon123/F5TTS-on-Pod.git
cd F5TTS-on-Pod
python -m venv venv

source venv/bin/activate
pip install git+https://github.com/VYNCX/F5-TTS-THAI.git

#จำเป็นต้องติดตั้งเพื่อใช้งานได้มีประสิทธิภาพกับ GPU เลือก Runpod version runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu2
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

# การใช้งาน เพื่อทำ Tect to Speech!
```sh
  python src/f5_tts/f5_tts_webui.py --share
```
![image_example](https://github.com/user-attachments/assets/467c9ec6-eb31-4a18-b295-82588edee991)

เคล็ดลับ :
- สามารถตั้งค่า "ตัวอักษรสูงสุดต่อส่วน" หรือ max_chars เพื่อลดความผิดพลาดการอ่าน แต่ความเร็วในการสร้างจะช้าลง สามารถปรับลด NFE Step เพื่อเพิ่มความเร็วได้.
- อย่าลืมเว้นวรรคประโยคเพื่อให้สามารถแบ่งส่วนในการสร้างได้.
- สำหรับ ref_text หรือ ข้อความตันฉบับ แนะนำให้ใช้เป็นภาษาไทยหรือคำอ่านภาษาไทยสำหรับเสียงภาษาอื่น เช่น Good Morning > กู้ดมอร์นิ่ง.
- สำหรับเสียงต้นแบบ ควรใช้ความยาวไม่เกิน 10 วินาที ถ้าเป็นไปได้ห้ามมีเสียงรบกวน.
  
# ฝึกอบรม และ Finetune
ใช้งานบน Google Colab [Finetune](https://colab.research.google.com/drive/1jwzw4Jn1qF8-F0o3TND68hLHdIqqgYEe?usp=sharing) หรือ 

ติดตั้ง

```sh
  cd F5-TTS-THAI
  pip install -e .
```

เปิด Gradio
```sh
  f5-tts_finetune-gradio
```

# ตัวอย่างเสียง

- เสียงต้นฉบับ
- ข้อความ : ได้รับข่าวคราวของเราที่จะหาที่มันเป็นไปที่จะจัดขึ้น.
  
https://github.com/user-attachments/assets/003c8a54-6f75-4456-907d-d28897e4c393

- เสียงที่สร้าง 1(ข้อความเดียวกัน)
- ข้อความ : ได้รับข่าวคราวของเราที่จะหาที่มันเป็นไปที่จะจัดขึ้น.
   
https://github.com/user-attachments/assets/926829f2-8d56-4f0f-8e2e-d73cfcecc511

- เสียงที่สร้าง 2(ข้อความใหม่)
- ข้อความ : ฉันชอบฟังเพลงขณะขับรถ เพราะช่วยให้รู้สึกผ่อนคลาย

https://github.com/user-attachments/assets/06d6e94b-5f83-4d69-99d1-ad19caa9792b

# อ้างอิง

- [F5-TTS](https://github.com/SWivid/F5-TTS)
- Thai Dataset : [Porameht/processed-voice-th-169k](https://huggingface.co/datasets/Porameht/processed-voice-th-169k)





