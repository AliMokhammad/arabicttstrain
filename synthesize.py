import ruamel.yaml
from model.models import ForwardTransformer
from pathlib import Path
from data.audio import Audio
from vocoding.predictors import HiFiGANPredictor

yaml = ruamel.yaml.YAML()


def synthesize(text):
    if not text:
        return None
    wav_no_vc_path = Path("./output/wav_no_vc")
    wave_vc_path = Path("./output/wav_vc")

    vocoder = HiFiGANPredictor.from_folder('vocoding/hifigan/custom/')
    # with open("./step_95000/config.yaml", 'rb') as session_yaml:
    #     config = yaml.load(session_yaml)
    # model = ForwardTransformer.from_config(config)
    model = ForwardTransformer.load_model('./step_95000')
    # audio = Audio(config=config)
    audio = Audio.from_config(model.config)
    out = model.predict(text, speed_regulator=1.)

    wav_no_vc = audio.reconstruct_waveform(out['mel'].numpy().T)
    wav_vc = vocoder([out['mel'].numpy().T])[0]

    audio.save_wav(wav_no_vc, wav_path=(wav_no_vc_path).with_suffix('.wav'))
    audio.save_wav(wav_vc, wav_path=(wave_vc_path).with_suffix('.wav'))


synthesize("لَقَدْ قَدِمْتُ الْبَصْرَةَ، فَأَصَبْتَ آلَافًا فَمَا اكْتَرَثْتُ بِهَا فَرَحًا، وَلا حَدَّثْتُ نَفْسِي بِالْكُرْهِ أَيْضًا")


# ثُمَّ مَضَى إِلَى الْيَمَنِ فَقَتَلَ بِهَا ابْنَيْ عُبَيْدِ اللَّهِ بْنِ عَبَّاسٍ، صَبِيَّيْنِ مُلَيْحَيْنِ، فَهَامَتْ أُمُّهُمَا بِهِمَا
# ضَرَبَ الزُّبَيْرُ أَسْمَاءَ، فَصَاحَتْ لِعَبْدِ اللَّهِ بْنِ الزُّبَيْرِ، فَأَقْبَلَ، فَلَمَّا رَآهُ، قَالَ: أُمَّكَ طَالِقٌ إِنْ دَخَلَتْ
# آخِرُ الْمُهَاجِرِينَ وَالْمُهَاجِرَاتِ وَفَاةً، وَأُمُّهَا قُتَيْلَةُ بِنْتُ عَبْدِ الْعُزَّى الْعَامِرِيَّةُ، لَهَا عِدَّةُ أَحَادِيثَ.
# فَأَرْسَلَ إِلَيْهَا لَتَأْتِيَنَّ أَوْ لَأَبْعَثَنَّ مَنْ يَسْحَبُكِ بِقُرُونِكِ
# وَقُتِلَ شُرَيْحُ بْنُ هَانِئٍ الْحَارِثِيُّ، وَأَصَابَ الْعَسْكَرَ ضِيقٌ وَجُوعٌ شَدِيدٌ، حَتَّى هَلَكَ عَامَّتُهُمْ
# لِأَنَّهُ يَتَلَذَّذُ بِأَنْوَاعِ الْمَأْكَلِ وَالْمَشْرَبِ وَغَيْرِ ذَلِكَ، أَيْ أَغْلَبُ أَفْرَادِهِ فَلَا يَرِدُ الْمَرِيضُ الَّذِي أَضْنَاهُ الْمَرَضُ فَصَارَ لَا يَقْدِرُ عَلَى تَنَاوُلِ مَا فِيهِ لَذَّةٌ أَوْ كُلُّ أَفْرَادِهِ، وَنَقُولُ: الْكَافِرُ الْمَذْكُورُ بِاعْتِبَارِ مَا يَعْقُبُهُ مِنْ أَنْوَاعِ الْأَلَمِ فِي الْآخِرَةِ.
# لَقَدْ قَدِمْتُ الْبَصْرَةَ، فَأَصَبْتَ آلَافًا فَمَا اكْتَرَثْتُ بِهَا فَرَحًا، وَلا حَدَّثْتُ نَفْسِي بِالْكُرْهِ أَيْضًا
# إِنِّي سَمِعْتُ أَبَا الدَّرْدَاءِ، يَقُولُ: إِنَّ ذَا الدِّرْهَمَيْنِ يَوْمَ الْقِيَامَةِ أَشَدُّ حِسَابًا مِنْ ذِي الدِّرْهَمِ
