class VCFReader():
  def __init__(self, filepath):
    self.fp = filepath

    self.vcf_file = open(self.fp, 'r')

    self.line = self.vcf_file.readline()
    while self.line.startswith('##'):
      self.line = self.vcf_file.readline()

    # Get sample names
    self.samples = self.line.split()[9:]

  def get_samples(self):
    return self.samples

  def __iter__(self):
    return self

  def __next__(self):
    self.line = self.vcf_file.readline()

    if not self.line:
      raise StopIteration

    toks = self.line.split()
    return (int(toks[1]), toks[9:])

  def close(self):
    if self.vcf_file:
      self.vcf_file.close()
      self.vcf_file = None

  def __del__(self):
    if self.vcf_file:
      self.close()
