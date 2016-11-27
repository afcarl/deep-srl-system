class Word(object):

    def __init__(self, elem, file_encoding='utf-8'):
        self.index = int(elem[0]) - 1
        self.form = elem[1].lower().decode(file_encoding)
        self.is_prd = self._set_is_prd(elem[12])
        self.labels = elem[14:]

    @staticmethod
    def _set_is_prd(prd):
        if prd is 'Y':
            return True
        return False
