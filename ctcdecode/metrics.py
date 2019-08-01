import editdistance


def wer(ref, hyp):
    ref = ref.split()
    hyp = hyp.split()
    return editdistance.eval(ref, hyp)


def cer(ref, hyp):
    ref = list(ref)
    hyp = list(hyp)

    return editdistance.eval(ref, hyp)
